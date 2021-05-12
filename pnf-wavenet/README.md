# Parallel and Flexible Sampling from Wavenet
![alt text](../images/super_res_panel.png)

### Summary
We can use a Wavenet generateive model as a prior to solve many useful tasks such as source separation, inpainting, and super resolution. Find the instructions below for each specific task. We provide pretrained models for voice data from the VCTK dataset, and piano data from the Supra Piano datset. This codebase is largely based on this [implementation of wavenet](https://github.com/r9y9/wavenet_vocoder)

## Getting Started
Make sure all the requirements in the requirements.txt are installed. We tested the code with torch 1.6 and cuda 10.0.

There are many pretrained models corresponding to the different noise levels that we use for annealing.

Download Pretrained Models: [Here](https://drive.google.com/drive/folders/1zC_WzbweJqd63RRoRGm8yXOkBiFp3vsZ?usp=sharing). If you're working in a command-line environment, we recommend using [gdown](https://github.com/wkentaro/gdown) to download the checkpoint files.


## Spectrogram Conditioned Generation
Download the checkpoints for the folder `mulaw_maxaudio_spectrogram` and place them in `pnf-wavenet/egs/mulaw_maxaudio_spectrogram/` while preserving the file structure.

Here is a sample command. The outputs will be writtien to the folder `spectrogram_conditioned_outputs` but you can change that.
```
CUDA_VISIBLE_DEVICES=0 python wavenet_vocoder/pnf/sgld_spectrogram_descent.py
    egs/mulaw_maxaudio_spectrogram/vctk
    egs/mulaw_maxaudio_spectrogram/vctk/dump_norm/dev
    spectrogram_conditioned_outputs
    --preset="egs/mulaw_maxaudio_spectrogram/config.json"
```

Creating spectrograms for input: If you look in `egs/mulaw_maxaudio_spectrogram/vctk/dump_norm/dev` you'll see an audio file and spectrogram that was used in the generation process. These spectrograms are generated with specific parameters and normalized. If you want to generate spectrograms that are compatible with the generation process, follow these steps:



## Super-resolution
Download the checkpoints for the folder `mulaw_maxaudio` and place them in `pnf-wavenet/egs/mulaw_maxaudio` while preserving the file structure.

Here is a sample command that will do 4x super resolution. The outputs will be written to the folder `superres_outputs` but you can change that. 
```
CUDA_VISIBLE_DEVICES=0 python wavenet_vocoder/pnf/sgld_superres.py
    egs/mulaw_maxaudio/supra_piano
    egs/sample_piano.wav
    superres_outputs
    --preset="egs/mulaw_maxaudio/config.json" 
    --downsample_interval 4
```

You can change the input audio file or the downsample interval for different results. The time it takes to run the algorithm will be dependent on the length of the file. Also, the higher the upsampling rate, the more iterations (N_STEPS) you will need for good results. For 8x or higher it is recommended to change N_STEPS to 512 or 1024.

## Inpainting
Download the checkpoints for the folder `mulaw_maxaudio` and place them in `pnf-wavenet/egs/mulaw_maxaudio` while preserving the file structure.

Here is a sample command that will do 200ms inpainting. The outputs will be written to the folder `inpainting_outputs` but you can change that. 
```
CUDA_VISIBLE_DEVICES=0 python wavenet_vocoder/pnf/sgld_inpainting.py
    egs/mulaw_maxaudio/supra_piano
    egs/sample_piano.wav
    inpainting_outputs
    --preset="egs/mulaw_maxaudio/config.json"
```
The inpainting gap is defined in the file as `GAP`. You can change this to inpaint a different part or length of the file. The longer the gap, the more iterations (N_STEPS) you will need for good results.

## Source Separation
Download the checkpoints for the folder `linear_quantize_max_audio` and place them in `pnf-wavenet/egs/linear_quantize_max_audio`. Source separation is the only task that does not work with a mu-law quantized wavenet because of the linearity assumption of source separation (mixture = source1 + source2).

Here is a sample command that separates a mixture of a piano and voice. The outputs will be written to the folder `separation_outputs` but you can change that.
```
CUDA_VISIBLE_DEVICES=0 python wavenet_vocoder/pnf/sgld_separation.py
    egs/linear_quantize_max_audio/supra_piano
    egs/linear_quantize_max_audio/vctk/
    egs/sample_piano.wav
    egs/sample_voice.wav
    separation_outputs 
    --preset="egs/linear_quantize_max_audio/config.json"
```

The arguments require one audio file of each source type which have the same length. If you want to feed a mixture directly, you can easily modify the python file to load the mixture instead of adding the two inputs. 

## Multi-GPU and HyperParameters
All code should will run with multi-gpus if you specify multiple gpus in CUDA_VISIBLE_DEVICES. You will also need to change BATCH_SIZE in the corresponding python file to be a multiple of the number of gpus. We have been using 2 * n_gpus which works well. If you try to parallelize over a an audio sequence that is too small, the concurrent updates will cause the optimization to blow up. Each update in the batch_size is grabbing a patch of size SGLD_WINDOW and updating it concurrently. Therefore, BATCH_SIZE * SGLD_WINDOW must be << than the total SAMPLE_SIZE of the audio that is given as input. You can specify SAMPLE_SIZE in each python file, -1 means use the entire audio file otherwise you can use a part of the audio file by setting that value. 

## Runtime and N_STEPS
The number of iterations required for good outputs is mostly based on the strength of the conditioning signal. If you are running 2x super-reslution or spectrogram conditioned generation, N_STEPS = 256 is a reasonable value. If you are running something less constrained like 16x super-resolution or inpainting with a large gap, the algorithm needs many more iterations to converge. N_STEPS = 1024 is a reasonable value in those cases. You can play around with the tradeoff between the runtime and quality of the output. 
