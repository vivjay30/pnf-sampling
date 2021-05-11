# coding: utf-8
"""
Synthesis waveform for testset

usage: separation.py [options] <checkpoint> <dump-root>

options:
    --hparams=<parmas>          Hyper parameters [default: ].
    --preset=<json>             Path of preset parameters (json).
    --steps_per_sigma=<steps>   Number of steps per sigma
    --seed=<seed>               Random Seed [default: 0].
    -h, --help                  Show help message.
"""
from docopt import docopt
import json
import os
from os.path import dirname, join, basename, splitext, exists
import string
import random

import time
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import audio
from hparams import hparams
from train import build_model, collate_fn, sanity_check
from evaluate import get_data_loader
from nnmnkwii import preprocessing as P
from wavenet_vocoder.util import linear_quantize, inv_linear_quantize
from wavenet_vocoder.pnf.pnf_utils import *


# optimization will blow up if BATCH_SIZE is too large relative to SAMPLE_SIZE/SGLD_WINDOW ratio
# because approximate independence of the gradient updates will be violated too often)
# could possibly try to fix this by carefully choosing non-overlapping update windows?
SAMPLE_SIZE = 256 * 200 # Each spectrogram frame is 256 samples
SGLD_WINDOW = 32000  # Must be divisible by 256
BATCH_SIZE = 2 # BATCH_SIZE % Number of gpus must be zero
N_STEPS = 256  # Langevin iterations per noise level

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def main(args):
    model = ModelWrapper()
    model.eval()

    receptive_field = model.receptive_field
    hparams.max_time_steps = SAMPLE_SIZE

    test_data_loader = get_data_loader(args["<dump-root>"], collate_fn)

    # Change the output dir if you want
    writing_dir = "spectrogram_conditioned_outputs"
    if not exists(writing_dir):
        os.makedirs(writing_dir)
    print("writing dir: {}".format(writing_dir))

    (x_original, y, c, g, input_lengths) = next(iter(test_data_loader))
    c = c.to(device)
    sanity_check(model.model, c, g)

    # Write inputs
    x_original_out = P.inv_mulaw_quantize(x_original, hparams.quantize_channels - 1)
    sf.write(join(writing_dir, "original.wav"), x_original_out[0, 0,], hparams.sample_rate)

    # Initialize with noise
    x = torch.FloatTensor(np.random.uniform(0, 256, size=(1, x_original.shape[-1] + 1))).to(device)
    x.requires_grad = True

    sigmas = [175.9, 110., 68.7,  42.9, 26.8, 16.8, 10.5, 6.55, 4.1, 2.56, 1.6, 1.0, 0.625, 0.39, 0.244, 0.1]

    t0 = time.time()

    for idx, sigma in enumerate(sigmas):
        # Bump down a model
        checkpoint_path = join(args["<checkpoint>"], CHECKPOINTS[sigma], "checkpoint_latest_ema.pth")
        model.load_checkpoint(checkpoint_path)
        parmodel = torch.nn.DataParallel(model)
        parmodel.to(device)

        eta = .1 * (sigma ** 2)

        # Make sure each sample is updated on average N_STEPS times
        n_steps_sgld = int((SAMPLE_SIZE/(SGLD_WINDOW*BATCH_SIZE)) * N_STEPS)
        print("Number of SGLD steps {}".format(n_steps_sgld))
        for i in range(n_steps_sgld):
            # Sample a random chunk of the spectrogram, accounting for padding
            # need to get a good sampling of the beginning/end (boundary effects)
            # to understand this: think about how often we would update x[0] (first point)
            # if we only sampled U(0,c.shape-receptive_field-SGLD_WINDOW)
            j = np.random.randint(hparams.cin_pad - SGLD_WINDOW // hparams.hop_size,
                                  c.shape[-1] - hparams.cin_pad, 
                                  BATCH_SIZE)
            j = np.maximum(j, hparams.cin_pad)
            j = np.minimum(j, c.shape[-1] - hparams.cin_pad - (SGLD_WINDOW // hparams.hop_size))
            # Get the corresponding start of the waveform
            x_start = (j - hparams.cin_pad) * hparams.hop_size

            patches_c = []
            patches_x = []
            for k in range(BATCH_SIZE):
                patches_c.append(c[0, :, j[k] - hparams.cin_pad : j[k] + hparams.cin_pad + (SGLD_WINDOW // hparams.hop_size)])
                patches_x.append(x[:, x_start[k] : x_start[k] + SGLD_WINDOW + 1])

            patches_c = torch.stack(patches_c, axis=0)
            patches_x = torch.stack(patches_x, axis=0)

            # Forward pass
            log_prob, prediction0 = parmodel(patches_x, c=patches_c, sigma=sigma)
            log_prob = torch.sum(log_prob)

            grad = torch.autograd.grad(log_prob, patches_x)[0]

            x_update = eta * grad

            # Langevin step
            epsilon = np.sqrt(2 * eta) * torch.normal(0, 1, size=x_update.shape, device=device)
            x_update += epsilon

            with torch.no_grad():
                for k in range(BATCH_SIZE):
                    x[:, x_start[k] : x_start[k] + SGLD_WINDOW + 1] += x_update[k]

            if (not i % 20) or (i == (n_steps_sgld - 1)): # debugging
                print("--------------")
                print("i {}".format(i))
                print("Max sample {}".format(
                  abs(x).max()))
                print('Mean sample logpx: {}'.format(log_prob / x.shape[-1]))
                print("Max gradient update: {}".format(eta * abs(grad).max()))


        out = P.inv_mulaw_quantize(x[0, 1:].detach().cpu().numpy(), hparams.quantize_channels - 1)
        out = np.clip(out, -1, 1)
        sf.write(join(writing_dir, "out_{}.wav".format(sigma)), out, hparams.sample_rate)

    final_time = time.time()
    with open(join(writing_dir, "info.json"), "w") as f:
        json.dump({"time": float(final_time - t0)}, f, indent=4)

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

    seed = int(args["--seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"

    main(args)
