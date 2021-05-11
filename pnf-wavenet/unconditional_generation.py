# coding: utf-8
"""
Synthesis waveform for testset

usage: unconditional_generation.py [options] <dump-root> <checkpoint> <dst_dir>

options:
    --hparams=<parmas>          Hyper parameters [default: ].
    --preset=<json>             Path of preset parameters (json).
    --length=<T>                Steps to generate [default: 32000].
    --speaker-id=<N>            Use specific speaker of data in case for multi-speaker datasets.
    --verbose=<level>           Verbosity level [default: 0].
    -h, --help                  Show help message.
"""
from docopt import docopt

import sys
from glob import glob
import os
from os.path import dirname, join, basename, splitext, exists
import torch
import numpy as np
import random
from tqdm import tqdm
from scipy.io import wavfile

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw, \
    is_linear_quantize, linear_quantize, inv_linear_quantize

import audio
from hparams import hparams

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def to_int16(x):
    if x.dtype == np.int16:
        return x
    assert x.dtype == np.float32
    assert x.min() >= -1 and x.max() <= 1.0
    return (x * 32767).astype(np.int16)


if __name__ == "__main__":
    args = docopt(__doc__)
    verbose = int(args["--verbose"])
    if verbose > 0:
        print("Command line args:\n", args)
    data_root = args["<dump-root>"]
    checkpoint_path = args["<checkpoint>"]
    dst_dir = args["<dst_dir>"]
    length = int(args["--length"])
    preset = args["--preset"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    else:
        hparams_json = join(dirname(checkpoint_path), "hparams.json")
        if exists(hparams_json):
            print("Loading hparams from {}".format(hparams_json))
            with open(hparams_json) as f:
                hparams.parse_json(f.read())

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"

    hparams.max_time_sec = None
    hparams.max_time_steps = None

    from train import build_model
    from synthesis import batch_wavegen

    # Model
    model = build_model().to(device)

    # Load checkpoint
    print("Load checkpoint from {}".format(checkpoint_path))
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])

    # Generate
    y_hat = batch_wavegen(model, c=None, g=None, fast=True,
                          tqdm=tqdm, length=length)
    gen = y_hat[0, :length]
    gen = np.clip(gen, -1.0, 1.0)

    # Write, random name
    os.makedirs(dst_dir, exist_ok=True)
    dst_wav_path = join(dst_dir, "{}_gen.wav".format(random.randint(0, 2**16)))
    wavfile.write(dst_wav_path, hparams.sample_rate, to_int16(gen))
