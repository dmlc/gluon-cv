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

|       model       | f32 latency(ms) | s8 latency(ms) | f32 throughput(fps, BS=64) | s8 throughput(fps, BS=64) | f32 accuracy | s8 accuracy |
|:-----------------:|:---------------:|:--------------:|:--------------------------:|:-------------------------:|:------------:|:-----------:|
|    resnet50_v1    |      11.88      |      2.45      |           197.37           |          1464.61          |  77.21/93.56 | 77.02/93.45 |
| resnet50_v1d_0.11 |       8.51      |      2.00      |           1099.79          |          10956.07         |  63.06/84.64 | 60.38/82.65 |
|    mobilenet1.0   |       4.23      |      0.93      |           549.15           |          5654.21          |  73.28/91.22 | 72.77/90.93 |
|  mobilenetv2_1.0  |      16.19      |      1.51      |           217.32           |          4961.84          |  71.89/90.53 | 71.69/90.44 |
|   squeezenet1.0   |       4.46      |      1.04      |           550.50           |          3532.40          |  57.74/80.33 | 54.75/78.30 |
|   squeezenet1.1   |       3.53      |      0.98      |           1008.26          |          6365.97          |  58.00/80.47 | 55.70/79.29 |
|    inceptionv3    |      18.20      |      4.83      |           152.73           |           963.44          |  78.80/94.37 | 78.74/94.28 |
|       vgg16       |      15.86      |      7.93      |            86.31           |           400.85          |  73.06/91.18 | 73.08/91.18 |

Please refer to [GluonCV Model Zoo](http://gluon-cv.mxnet.io/model_zoo/index.html#image-classification)
for available pretrained models, training hyper-parameters, etc.
