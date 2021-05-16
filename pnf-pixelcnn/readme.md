## PixelCNN++ PnF Sampling

PixelCNN++ model taken from Lucas Caccia's PyTorch [implementation.](https://github.com/pclucas14/pixel-cnn-pp)

Pre-trained (unsmoothed) model is available [here](https://mega.nz/#F!W7IhST7R!PV7Pbet8Q07GxVLGnmQrZg)

### Running the training code

Finetune a smoothed model at a given NOISE\_LEVEL beginning from a pretrained checkpoint.

```
python main.py --load_params pretrained/pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth --lr .00040 -g NOISE_LEVEL -x 10
```

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


