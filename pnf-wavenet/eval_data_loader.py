import glob
import os
import random
import math

from pathlib import Path
from hparams import hparams

import numpy as np

from nnmnkwii import preprocessing as P
from wavenet_vocoder.util import linear_quantize, inv_linear_quantize

import librosa

"""
QUANTIZE TYPES
0: Mulaw
1: Linear
"""

def get_piano_file(idx, duration, quantize_type):
    """
    Gets one of the test supra piano samples
    """
    BASE_PATH = "/projects/grail/audiovisual/datasets/supra-rw-mp3/test"

    file_list = list(Path(BASE_PATH).rglob("*.mp3"))
    curr_file = random.choice(file_list)
    y, sr = librosa.core.load(curr_file, sr=22050)
    y /= abs(y).max()

    num_samples = y.shape[0]
    start_idx = random.randint(0, num_samples - duration)
    y = y[start_idx: start_idx + duration]
   

    # Mulaw, linear or linear max audio
    if quantize_type == 0:
        quantized = P.mulaw_quantize(y, hparams.quantize_channels - 1)

    elif quantize_type == 1:
        quantized = linear_quantize(y, hparams.quantize_channels - 1)

    return quantized


def get_voice_file(idx, duration, quantize_type):
    """
    Gets one of the last VCTK voices
    """
    BASE_PATH = "/projects/grail/audiovisual/datasets/VCTK-Corpus/wav48/test"

    assert(idx in list(range(0, 100)))
    if idx < 25:
        speaker_path = os.path.join(BASE_PATH, "p345")
    elif idx < 50:
        speaker_path = os.path.join(BASE_PATH, "p361")
    elif idx < 75:
        speaker_path = os.path.join(BASE_PATH, "p362")
    elif idx < 100:
        speaker_path = os.path.join(BASE_PATH, "p374")

    file_list = list(Path(speaker_path).rglob('*.wav'))

    curr_file = random.choice(file_list)
    y, sr = librosa.core.load(curr_file, sr=22050)

    y /= abs(y).max()
    start_idx = len(y) // 2
    y = y[int(start_idx - duration / 2): int(start_idx + duration / 2)]
    

    # Mulaw, linear or linear max audio
    if quantize_type == 0:
        quantized = P.mulaw_quantize(y, hparams.quantize_channels - 1)

    elif quantize_type == 1:
        quantized = linear_quantize(y, hparams.quantize_channels - 1)

    return quantized


def si_sdr(estimated_signal, reference_signals, scaling=True):
    """
    This is a scale invariant SDR. See https://arxiv.org/pdf/1811.02508.pdf
    or https://github.com/sigsep/bsseval/issues/3 for the motivation and
    explanation
    Input:
        estimated_signal and reference signals are (N,) numpy arrays
    Returns: SI-SDR as scalar
    """
    Rss = np.dot(reference_signals, reference_signals)
    this_s = reference_signals

    if scaling:
        # get the scaling factor for clean sources
        a = np.dot(this_s, estimated_signal) / Rss
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()

    SDR = 10 * math.log10(Sss/Snn)

    return SDR
