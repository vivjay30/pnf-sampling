## PixelCNN++ PnF Sampling

PixelCNN++ model taken from Lucas Caccia's PyTorch [implementation](https://github.com/pclucas14/pixel-cnn-pp).

Pre-trained (unsmoothed) model is available [here](https://mega.nz/#F!W7IhST7R!PV7Pbet8Q07GxVLGnmQrZg). Extract and store this model in the './pretrained/' subdirectory.

Pre-trained finetuned (smoothed) PixelCNN++ models are available [here](https://drive.google.com/uc?export=download&id=1HSyjV1ntjd6gEilUprwuAxjgNVhxOsmY) (~3.5Gb). Extract and store these models in the './finetuned' subdirectory.

### Running source separation

Source separation given a directory containing finetuned smoothed models.

```
python3 sep.py -r finetuned
```

### Running super-resolution 

Super-resolution given a directory containing finetuned smoothed models.

```
python3 super.py -r finetuned
```

### Running inpainting 

Inpainting given a directory containing finetuned smoothed models.

```
python3 inpaint.py -r finetuned
```

### Running the training code
 
Finetune a smoothed model at a given NOISE\_LEVEL beginning from a pretrained checkpoint.

```
python main.py --load_params pretrained/pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth --lr .00040 -g NOISE_LEVEL -x 10
```

Finetuning should not be necessary if you use the pre-trained finetuned models (provided above) but we provide this script for reproducibility of those finetuned models.
