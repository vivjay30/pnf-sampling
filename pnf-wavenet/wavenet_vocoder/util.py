# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import numpy as np

def _assert_valid_input_type(s):
    assert s == "mulaw-quantize" or s == "mulaw" or s == "raw" or s == "linear-quantize"


def is_mulaw_quantize(s):
    _assert_valid_input_type(s)
    return s == "mulaw-quantize"

def is_linear_quantize(s):
    _assert_valid_input_type(s)
    return s == "linear-quantize"


def is_mulaw(s):
    _assert_valid_input_type(s)
    return s == "mulaw"


def is_raw(s):
    _assert_valid_input_type(s)
    return s == "raw"


def is_scalar_input(s):
    return is_raw(s) or is_mulaw(s)


def linear_quantize(audio, quantization_channels):
    encoded = np.round(((audio + 1) * (float(quantization_channels))) / 2)
    return encoded


def inv_linear_quantize(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    return ((output * 2 / (float(quantization_channels))) - 1.0)
