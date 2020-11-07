# Image Classification on ImageNet

## Inference/Calibration Tutorial

### Float32 Inference

```
python verify_pretrained.py --model=resnet50_v1d_0.11 --batch-size=1
```

### Calibration

Naive calibrate model by using 5 batch data (32 images per batch). Quantized model will be saved into `./model/`.

```
python verify_pretrained.py --model=resnet50_v1d_0.11 --batch-size=32 --calibration
```

### INT8 Inference

```
python verify_pretrained.py --model=resnet50_v1d_0.11 --batch-size=1 --deploy --model-prefix=./model/resnet50_v1d_0.11-quantized-naive
```

## Performance

model | f32 latency(ms) | s8 latency(ms) | f32 throughput(fps, BS=64) | s8 throughput(fps, BS=64) | f32 accuracy | s8 accuracy
-- | -- | -- | -- | -- | -- | --
resnet50_v1 | 11.36 | 2.54 | 190.2 | 1363.75 | 77.21/93.56 | 76.34/93.13
resnet50_v1d_0.11 | 8.84 | 1.74 | 1070.66 | 10686.77 | 63.06/84.64 | 62.68/84.43
mobilenet1.0 | 3.88 | 0.88 | 583.05 | 5615.58 | 73.28/91.22 | 72.23/90.64
mobilenetv2_1.0 | 18.10 | 1.34 | 226.27 | 5005.94 | 71.89/90.53 | 70.87/89.88
squeezenet1.0 | 4.18 | 0.96 | 590.76 | 3393.09 | 57.74/80.33 | 56.98/79.66
squeezenet1.1 | 3.31 | 0.87 | 964.83 | 6027.15 | 58.00/80.47 | 57.02/79.73
inceptionv3 | 20.73 | 4.99 | 156.63 | 917.67 | 78.80/94.37 | 77.36/93.57
vgg16 | 16.71 | 7.63 | 87.17 | 399.62 | 73.06/91.18 | 71.94/90.59

Please refer to [GluonCV Model Zoo](http://gluon-cv.mxnet.io/model_zoo/index.html#image-classification)
for available pretrained models, training hyper-parameters, etc.
