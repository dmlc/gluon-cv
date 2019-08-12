# Single Shot Multibox Object Detection [1]

[GluonCV Model Zoo](http://gluon-cv.mxnet.io/model_zoo/index.html#object-detection)

- `--dali` Use [DALI](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html) for faster data loading and data preprocessing in training with COCO dataset. DALI >= 0.12 required.
- `--amp` Use [Automatic Mixed Precision training](https://mxnet.incubator.apache.org/versions/master/tutorials/amp/amp_tutorial.html), automatically casting FP16 where safe.
- `--horovod` Use [Horovod](https://github.com/horovod/horovod) for distributed training, with a network agnostic wrapper for the optimizer, allowing efficient allreduce using OpemMPI and NCCL.

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

model | f32 latency(ms) | s8 latency(ms) | f32 throughput(fps, BS=256) | s8 throughput(fps, BS=256) | f32 accuracy | s8 accuracy
-- | -- | -- | -- | -- | -- | --
ssd_300_vgg16_atrous_voc | 105.60 | 13.08 | 19.47 | 110.14 | 77.49 | 77.49
ssd_512_vgg16_atrous_voc | 215.05 | 32.63 | 6.76 | 36.56 | 78.82 | 78.82
ssd_512_mobilenet1.0_voc | 28.98 | 6.97 | 65.55 | 210.17 | 75.51 | 75.49
ssd_512_resnet50_v1_voc | 52.77 | 11.75 | 28.68 | 143.61 | 80.24 | 80.23

## References
1. Wei Liu, et al. "SSD: Single shot multibox detector" ECCV 2016.
2. Cheng-Yang Fu, et al. "[DSSD : Deconvolutional Single Shot Detector](https://arxiv.org/abs/1701.06659)" arXiv 2017.
