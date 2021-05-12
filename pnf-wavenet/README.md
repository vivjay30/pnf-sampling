# Parallel and Flexible Sampling from Wavenet
![alt text](../images/super_res_panel.png)

### Summary
We can use a Wavenet generateive model as a prior to solve many useful tasks such as source separation, inpainting, and super resolution. Find the instructions below for each specific task. We provide pretrained models for voice data from the VCTK dataset, and piano data from the Supra Piano datset. This codebase is largely based on this [implementation of wavenet](https://github.com/r9y9/wavenet_vocoder)

## Getting Started
Make sure all the requirements in the requirements.txt are installed. We tested the code with torch 1.6 and cuda 10.0.

There are many pretrained models corresponding to the different noise levels that we use for annealing.

Download Pretrained Models: [Here](https://drive.google.com/drive/folders/1YeuHPvqmaPMGvcSOb9J-hnLDYSbK1S2c?usp=sharing). If you're working in a command-line environment, we recommend using [gdown](https://github.com/wkentaro/gdown) to download the checkpoint files.


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


You can easily produce results like those in our demo videos. Our pre-trained real models work with the 4 mic [Seed ReSpeaker MicArray v 2.0](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/). We even provide a sample 4 channel file for you to run [Here](https://drive.google.com/drive/folders/1YeuHPvqmaPMGvcSOb9J-hnLDYSbK1S2c?usp=sharing). When you capture the data, it must be a m channel recording. Run the full command like below. For moving sources, reduce the duration flag to 1.5 and add `--moving` to stop the search at a coarse window.
```
python cos/inference/separation_by_localization.py \
    /path/to/model.pt \
    /path/to/input_file.wav \
    outputs/some_dirname/ \
    --n_channels 4 \
    --sr 44100 \
    --mic_radius .03231 \
    --use_cuda
```

## Rendering Synthetic Spatial Data
For training and evaluation, we use synthetically rendered spatial data. We place the voices in a virtual room and render the arrival times, level differences, and reverb using pyroomacoustics. We used the VCTK dataset but any voice dataset would work. An example command is below
```
python cos/generate_dataset.py \
    /path/to/VCTK/data \
    ./outputs/somename \
    --input_background_path any_bg_audio.wav \
    --n_voices 2 \
    --n_outputs 1000 \
    --mic_radius {radius} \
    --n_mics {M}
```

## Training on Synthetic Data
Below is an example command to train on the rendered data. You need to replace the training and testing dirs with the path to the generated datasets from above. We highly recommend initializing with a pre-trained model (even if the number of mics is different) and not training from scratch.
```
python cos/training/train.py \
   ./generated/train_dir \
   ./generated/test_dir \
   --name experiment_name \
   --checkpoints_dir ./checkpoints \
   --pretrain_path ./path/to/pretrained.pt \
   --batch_size 8 \
   --mic_radius {radius} \
   --n_mics {M} \
   --use_cuda
```
__Note__: The training code expects you to have `sox` installed. The easiest way to install is to install it using conda as follows: `conda install -c conda-forge -y sox`.

***

## Training on Real Data
For those looking to improve on the pretrained models, we recommend gathering a lot more real data. We did not have the ability to gather very accurately positioned real data in a proper sound chamber. By training with a lot more real data, the results will almost certainly improve. All you have to do is create synthetic composites of speakers in the same format as the synthetic data, and run the same training script.

## Evaluation
For the synthetic data and evaluation, we use a setup of 6 mics in a circle of radius 7.25 cm. The following is instructions to obtain results on mixtures of N voices and no backgrounds. First generate a synthetic datset with the microphone setup specified previous with ```--n_voices 8``` from the test set of VCTK. Then run the following script:  

```
python cos/inference/evaluate_synthetic.py \
    /path/to/rendered_data/ \
    /path/to/model.pt \
    --n_channels 6 \
    --mic_radius .0725 \
    --sr 44100 \
    --use_cuda \
    --n_workers 1 \
    --n_voices {N}
```

Add ```--prec_recall``` separately to get the precision and recall.

| Number of Speakers N | 2     | 3     | 4     | 5     | 6     | 7     | 8     |
|----------------------|-------|-------|-------|-------|-------|-------|-------|
| Median SI-SDRi (dB)  | 13.9  | 13.2  | 12.2  | 10.8  | 9.1   | 7.2   | 6.3   |
| Median Angular Error | 2.0   | 2.3   | 2.7   | 3.5   | 4.4   | 5.2   | 6.3   |
| Precision            | 0.947 | 0.936 | 0.897 | 0.912 | 0.932 | 0.936 | 0.966 |
| Recall               | 0.979 | 0.972 | 0.915 | 0.898 | 0.859 | 0825  | 0.785 |


