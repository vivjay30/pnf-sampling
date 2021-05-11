# coding: utf-8
"""
Synthesis waveform for testset

usage: separation.py [options] <checkpoint> <input-file>

options:
    --hparams=<parmas>          Hyper parameters [default: ].
    --preset=<json>             Path of preset parameters (json).
    --downsample_interval=<int> For super-resolution, set this param as an int (e.g. 16)
    -h, --help                  Show help message.
"""
from docopt import docopt
import os
from os.path import dirname, join, basename, splitext, exists

import librosa
import time
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import audio
from hparams import hparams
from train import build_model
from nnmnkwii import preprocessing as P
from wavenet_vocoder.util import linear_quantize, inv_linear_quantize
from wavenet_vocoder.pnf.pnf_utils import *

# optimization will blow up if BATCH_SIZE is too large relative to SAMPLE_SIZE/SGLD_WINDOW ratio
# because approximate independence of the gradient updates will be violated too often)
# could possibly try to fix this by carefully choosing non-overlapping update windows?

# Total number of samples to use for the input and output
SAMPLE_SIZE = -1  # Set this to -1 to just process the entire input file
SGLD_WINDOW = 20000
BATCH_SIZE = 2  # BATCH_SIZE % Number of gpus must be zero
N_STEPS = 1024 # Langevin iterations per noise level

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def main(args):
    model = ModelWrapper()
    model.eval()

    downsample_interval = int(args["--downsample_interval"])

    receptive_field = model.receptive_field

    # Change the output dir if you want
    writing_dir = "super_resolution_outputs"
    os.makedirs(writing_dir, exist_ok=True)
    print("writing dir: {}".format(writing_dir))

    # Load up a samples
    x_original = librosa.core.load(args["<input-file>"], sr=hparams.sample_rate, mono=True)[0]

    # Hacky way to allow processing some or all of the file
    global SAMPLE_SIZE
    if SAMPLE_SIZE == -1:
        SAMPLE_SIZE = x_original.shape[0]

    x_original = x_original[:SAMPLE_SIZE]

    # Normalize to reduce encoding artifacts
    x_original /= abs(x_original).max()
    sf.write(os.path.join(writing_dir, "x_original.wav"), x_original, hparams.sample_rate)

    # Cut the sampling rate
    x_modified = x_original[::downsample_interval]
    x_modified_out = librosa.core.resample(x_modified,
                                           int(hparams.sample_rate / downsample_interval), hparams.sample_rate)
    sf.write(join(writing_dir, "x_modified.wav"), x_modified_out, hparams.sample_rate)
    x_modified = P.mulaw_quantize(x_modified, hparams.quantize_channels - 1)

    # Update constraint mask for super resolution. Masked spots don't update
    mask = np.ones_like(x_original)
    mask[::downsample_interval] = 0
    mask = torch.Tensor(mask).unsqueeze(0).to(device)

    # Initialize with noise for the samples we need to fill in, or x original for the samples
    # we are allowed to use
    noise = np.random.uniform(0, 256, size=x_original.shape)
    x = P.mulaw_quantize(x_original, hparams.quantize_channels - 1) * (1 - mask) + noise * (mask)
    x = torch.FloatTensor(x).unsqueeze(0).to(device)
    x.requires_grad = True

    sigmas = [175.9, 110., 68.7, 54.3, 42.9, 34.0, 26.8, 21.2, 16.8, 13.3, 10.5, 8.29, 6.55, 5.18, 4.1, 3.24, 2.56, 1.6, 1.0, 0.625, 0.39, 0.244, 0.15, 0.1]

    for idx, sigma in enumerate(sigmas):
        # Make sure each sample is updated on average N_STEPS times
        n_steps_sgld = int((SAMPLE_SIZE/(SGLD_WINDOW*BATCH_SIZE)) * N_STEPS)
        print("Number of SGLD steps {}".format(n_steps_sgld))
        
        # Bump down a model
        checkpoint_path = join(args["<checkpoint>"], checkpoints[sigma], "checkpoint_latest.pth")
        model.load_checkpoint(checkpoint_path)

        parmodel = torch.nn.DataParallel(model)
        parmodel.to(device)

        eta = .05 * (sigma ** 2)

        for i in range(n_steps_sgld):
            # need to get a good sampling of the beginning/end (boundary effects)
            # to understand this: think about how often we would update x[receptive_field] (first point)
            # if we only sampled U(receptive_field,x0.shape-receptive_field-SGLD_WINDOW)
            j = np.random.randint(-SGLD_WINDOW, x.shape[1], BATCH_SIZE)
            j = np.maximum(j, 0)
            j = np.minimum(j, x.shape[1]-(SGLD_WINDOW))

            patches = []
            for k in range(BATCH_SIZE):
                patches.append(x[:, j[k]:j[k] + SGLD_WINDOW])

            patches = torch.stack(patches, axis=0)

            # Forward pass
            log_prob, prediction = parmodel(patches, sigma=sigma)
            log_prob = torch.sum(log_prob)
            grad = torch.autograd.grad(log_prob, patches)[0]

            x_update = eta * grad

            # Langevin step
            epsilon = np.sqrt(2 * eta) * torch.normal(0, 1, size=x_update.shape, device=device)
            x_update += epsilon

            with torch.no_grad():
                for k in range(BATCH_SIZE):
                    x_update[k] *= mask[:, j[k] : j[k] + SGLD_WINDOW]
                    x[:, j[k] : j[k] + SGLD_WINDOW] += x_update[k]

            if (not i % 20) or (i == (n_steps_sgld - 1)): # debugging
                print("--------------")
                print('sigma = {}'.format(sigma))
                print('eta = {}'.format(eta))
                print("i {}".format(i))
                print("Max sample {}".format(
                    abs(x).max()))
                print('Mean sample logpx: {}'.format(log_prob / (BATCH_SIZE*SGLD_WINDOW)))
                print("Max gradient update: {}".format(eta * abs(grad).max()))
                t0 = time.time()


        out = P.inv_mulaw_quantize(x[0].detach().cpu().numpy(), hparams.quantize_channels - 1)
        out = np.clip(out, -1, 1)
        sf.write(os.path.join(writing_dir, "out_{}.wav".format(sigma)), out, hparams.sample_rate)

if __name__ == "__main__":
    args = docopt(__doc__)

    # Load preset if specified
    if args["--preset"] is not None:
        with open(args["--preset"]) as f:
            hparams.parse_json(f.read())
    else:
        hparams_json = join(dirname(args["<checkpoint1>"]), "hparams.json")
        if exists(hparams_json):
            print("Loading hparams from {}".format(hparams_json))
            with open(hparams_json) as f:
                hparams.parse_json(f.read())

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"

    main(args)
