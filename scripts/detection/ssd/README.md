# Single Shot Multibox Object Detection [1]

[GluonCV Model Zoo](http://gluon-cv.mxnet.io/model_zoo/index.html#object-detection)

## Inference/Calibration Tutorial

### Float32 Inference

```
python eval_ssd.py --network=mobilenet1.0 --data-shape=512 --batch-size=1
```

### Calibration

Naive calibrate model by using 5 batch data (32 images per batch). Quantized model will be saved into `./model/`.

```
python eval_ssd.py --network=mobilenet1.0 --data-shape=512 --batch-size=32 --calibration
```

### INT8 Inference

```
python eval_ssd.py --network=mobilenet1.0 --data-shape=512 --batch-size=1 --deploy --model-prefix=./model/ssd_512_mobilenet1.0_voc-quantized-naive
```

## Performance

|           model          | f32 latency(ms) | s8 latency(ms) | f32 throughput(fps, BS=256) | s8 throughput(fps, BS=256) | f32 accuracy | s8 accuracy |
|:------------------------:|:---------------:|:--------------:|:---------------------------:|:--------------------------:|:------------:|:-----------:|
| ssd_300_vgg16_atrous_voc |      62.57      |      13.11     |            20.26            |           110.01           |     77.49    |    77.33    |
| ssd_512_vgg16_atrous_voc |      166.38     |      30.30     |             7.72            |            37.71           |     78.82    |    78.70    |
| ssd_512_mobilenet1.0_voc |      28.20      |      7.79      |            62.87            |           192.98           |     75.51    |    74.78    |
|  ssd_512_resnet50_v1_voc |      49.80      |      12.71     |            28.83            |           138.74           |     80.24    |    80.28    |
| ssd_512_resnet101_v2_voc |      78.43      |      39.90     |            19.05            |            26.41           |     79.70    |    78.34    |

## References
1. Wei Liu, et al. "SSD: Single shot multibox detector" ECCV 2016.
2. Cheng-Yang Fu, et al. "[DSSD : Deconvolutional Single Shot Detector](https://arxiv.org/abs/1701.06659)" arXiv 2017.
