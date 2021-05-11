import librosa
import numpy as np
from glob import glob
import os

from wavenet_vocoder.util import inv_linear_quantize

in_dir = "/projects/grail/vjayaram/wavenet_vocoder/egs/linear_quantize/drums/dump/train_no_dev/"
extension = "*-wave.npy"
src_files = sorted(glob(os.path.join(in_dir, "**/") + extension, recursive=True))

all_data = []
for file_name in src_files:
    data = np.load(file_name)
    data = inv_linear_quantize(data, 255)
    all_data.append(np.abs(data))
    print(np.percentile(np.concatenate(all_data).flatten(), 99.5))
