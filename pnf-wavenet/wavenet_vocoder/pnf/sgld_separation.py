# coding: utf-8
"""
Synthesis waveform for testset

usage: separation.py [options] <checkpoint0> <checkpoint1> <input-file1> <input-file2> <output-dir>

options:
    --hparams=<parmas>          Hyper parameters [default: ].
    --preset=<json>             Path of preset parameters (json).
    -h, --help                  Show help message.
"""
from docopt import docopt
import json
import os
from os.path import dirname, join, basename, splitext, exists

import time
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import librosa

import audio
from hparams import hparams
from train import build_model
from nnmnkwii import preprocessing as P
from wavenet_vocoder.util import linear_quantize, inv_linear_quantize
from wavenet_vocoder.pnf.pnf_utils import *

# optimization will blow up if BATCH_SIZE is too large relative to SAMPLE_SIZE/SGLD_WINDOW ratio
# because approximate independence of the gradient updates will be violated too often)
# could possibly try to fix this by carefully choosing non-overlapping update windows?
SAMPLE_SIZE = -1 # -1 means process the whole file. Otherwise specify number of samples
SGLD_WINDOW = 20000
BATCH_SIZE = 2  # Must be divisible by the GPUs
N_STEPS = 256  # Increase for better quality outputs

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def main(args):
    model0 = ModelWrapper()
    model1 = ModelWrapper()
    
    receptive_field = model0.receptive_field

    writing_dir = args["<output-dir>"]
    os.makedirs(writing_dir, exist_ok=True)
    print("writing dir: {}".format(writing_dir))

    source1 = librosa.core.load(args["<input-file1>"], sr=22050, mono=True)[0]
    source2 = librosa.core.load(args["<input-file2>"], sr=22050, mono=True)[0]
    mixed = source1 + source2

    # Increase the volume of the mixture fo avoid artifacts from linear encoding
    mixed /= abs(mixed).max()
    mixed *= 1.4
    mixed = linear_quantize(mixed + 1.0, hparams.quantize_channels - 1)
    global SAMPLE_SIZE
    if SAMPLE_SIZE == -1:
        SAMPLE_SIZE = int(mixed.shape[0])

    mixed = mixed[:SAMPLE_SIZE]

    mixed = torch.FloatTensor(mixed).reshape(1, -1).to(device)

    # Write inputs
    mixed_out = inv_linear_quantize(mixed[0].detach().cpu().numpy(), hparams.quantize_channels - 1) - 1.0
    mixed_out = np.clip(mixed_out, -1, 1)
    sf.write(join(writing_dir, "mixed.wav"), mixed_out, hparams.sample_rate)

    # Initialize with noise
    x0 = torch.FloatTensor(np.random.uniform(0, 512, size=(1, SAMPLE_SIZE))).to(device)
    x0[:] = mixed - 127.0
    x0 = F.pad(x0, (receptive_field, receptive_field), "constant", 127)
    x0.requires_grad = True
    
    x1 = torch.FloatTensor(np.random.uniform(0, 512, size=(1, SAMPLE_SIZE))).to(device)
    x1[:] = 127.
    x1 = F.pad(x1, (receptive_field, receptive_field), "constant", 127)
    x1.requires_grad = True

    sigmas = [175.9, 110., 68.7, 54.3, 42.9, 34.0, 26.8, 21.2, 16.8, 13.3, 10.5, 8.29, 6.55, 5.18, 4.1, 3.24, 2.56, 1.6, 1.0, 0.625, 0.39, 0.244, 0.15, 0.1]

    np.random.seed(999)

    for idx, sigma in enumerate(sigmas):
        # We make sure each sample is updated a certain number of times
        n_steps = int((SAMPLE_SIZE/(SGLD_WINDOW*BATCH_SIZE))*N_STEPS)
        print("Number of SGLD steps {}".format(n_steps))
        # Bump down a model
        checkpoint_path0 = join(args["<checkpoint0>"], CHECKPOINTS[sigma], "checkpoint_latest_ema.pth")
        model0.load_checkpoint(checkpoint_path0)
        checkpoint_path1 = join(args["<checkpoint1>"], CHECKPOINTS[sigma], "checkpoint_latest_ema.pth")
        model1.load_checkpoint(checkpoint_path1)

        parmodel0 = torch.nn.DataParallel(model0)
        parmodel0.to(device)
        parmodel1 = torch.nn.DataParallel(model1)
        parmodel1.to(device)

        eta = .05 * (sigma ** 2)
        gamma = 15 * (1.0 / sigma) ** 2

        t0 = time.time()
        for i in range(n_steps):
            # need to get a good sampling of the beginning/end (boundary effects)
            # to understand this: think about how often we would update x[receptive_field] (first point)
            # if we only sampled U(receptive_field,x0.shape-receptive_field-SGLD_WINDOW)
            j = np.random.randint(receptive_field-SGLD_WINDOW, x0.shape[1]-receptive_field, BATCH_SIZE)
            j = np.maximum(j, receptive_field)
            j = np.minimum(j, x0.shape[1]-(SGLD_WINDOW+receptive_field))

            # Seed with noised up silence
            x0[0, :receptive_field] = torch.FloatTensor(np.random.normal(127, sigma, mixed[0, :receptive_field].shape)).to(device)
            x0[0, -receptive_field:] = torch.FloatTensor(np.random.normal(127, sigma, mixed[0, -receptive_field:].shape)).to(device)
            x1[0, :receptive_field] = torch.FloatTensor(np.random.normal(127, sigma, mixed[0, :receptive_field].shape)).to(device)
            x1[0, -receptive_field:] = torch.FloatTensor(np.random.normal(127, sigma, mixed[0, -receptive_field:].shape)).to(device)

            patches0 = []
            patches1 = []
            mixpatch = []
            for k in range(BATCH_SIZE):
                patches0.append(x0[:, j[k]-receptive_field:j[k]+SGLD_WINDOW+receptive_field])
                patches1.append(x1[:, j[k]-receptive_field:j[k]+SGLD_WINDOW+receptive_field])
                mixpatch.append(mixed[:, j[k]-receptive_field:j[k]-receptive_field+SGLD_WINDOW])

            patches0 = torch.stack(patches0,axis=0)
            patches1 = torch.stack(patches1,axis=0)
            mixpatch = torch.stack(mixpatch,axis=0)

            # Forward pass
            log_prob, prediction0 = parmodel0(patches0, sigma=sigma)
            log_prob0 = torch.sum(log_prob)
            grad0 = torch.autograd.grad(log_prob0, x0)[0]

            log_prob, prediction1 = parmodel1(patches1, sigma=sigma)
            log_prob1 = torch.sum(log_prob)
            grad1 = torch.autograd.grad(log_prob1, x1)[0]

            x0_update, x1_update = [], []
            for k in range(BATCH_SIZE): 
                x0_update.append(eta * grad0[:, j[k]:j[k]+SGLD_WINDOW])
                x1_update.append(eta * grad1[:, j[k]:j[k]+SGLD_WINDOW])

            # Langevin step
            for k in range(BATCH_SIZE):
                epsilon0 = np.sqrt(2 * eta) * torch.normal(0, 1, size=(1, SGLD_WINDOW), device=device)
                x0_update[k] += epsilon0

                epsilon1 = np.sqrt(2 * eta) * torch.normal(0, 1, size=(1, SGLD_WINDOW), device=device)
                x1_update[k] += epsilon1

            # Reconstruction step
            for k in range(BATCH_SIZE):
                x0_update[k] -= eta * gamma * (patches0[k][:,receptive_field:receptive_field+SGLD_WINDOW] + patches1[k][:,receptive_field:receptive_field+SGLD_WINDOW] - mixpatch[k])
                x1_update[k] -= eta * gamma * (patches0[k][:,receptive_field:receptive_field+SGLD_WINDOW] + patches1[k][:,receptive_field:receptive_field+SGLD_WINDOW] - mixpatch[k])

            with torch.no_grad():
                for k in range(BATCH_SIZE):
                    x0[:, j[k]:j[k]+SGLD_WINDOW] += x0_update[k]
                    x1[:, j[k]:j[k]+SGLD_WINDOW] += x1_update[k]

            if (not i % 40) or (i == (n_steps - 1)): # debugging
                print("--------------")
                print('sigma = {}'.format(sigma))
                print('eta = {}'.format(eta))
                print("i {}".format(i))
                print("Max sample {}".format(
                    abs(x0).max()))
                print('Mean sample logpx: {}'.format(log_prob0 / (BATCH_SIZE*SGLD_WINDOW)))
                print('Mean sample logpy: {}'.format(log_prob1 / (BATCH_SIZE*SGLD_WINDOW)))
                print("Max gradient update: {}".format(eta * abs(grad0).max()))
                print("Reconstruction: {}".format(abs(x0[:, receptive_field:-receptive_field] + x1[:, receptive_field:-receptive_field] - mixed).mean()))
                print('Elapsed time = {}'.format(time.time()-t0))
                t0 = time.time()


        out0 = inv_linear_quantize(x0[0].detach().cpu().numpy(), hparams.quantize_channels - 1)
        out0 = np.clip(out0, -1, 1)
        sf.write(join(writing_dir, "out0_{}.wav".format(sigma)), out0, hparams.sample_rate)

        out1 = inv_linear_quantize(x1[0].detach().cpu().numpy(), hparams.quantize_channels - 1)
        out1 = np.clip(out1, -1, 1)
        sf.write(join(writing_dir, "out1_{}.wav".format(sigma)), out1, hparams.sample_rate)


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
