# Semantic Segmentation

## Inference/Calibration Tutorial

### FP32 inference

```
export OMP_NUM_THREADS=$(vCPUs/2)
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

# dummy data
python test.py --model=fcn --backbone=resnet101 --eval --batch-size=1 --pretrained --benchmark

# real data
python test.py --model=fcn --backbone=resnet101 --eval --batch-size=1 --pretrained
```

### Calibration
Naive calibrated model by using 5 batch data (32 images per batch). Quantized model will be saved into `./model/`.

```
# pascal-voc
python test.py --model=fcn --backbone=resnet101 --eval --batch-size=1 --pretrained --calibration --dataset=voc

# coco
python test.py --model=fcn --backbone=resnet101 --eval --batch-size=1 --pretrained --calibration --dataset=coco
```

### INT8 Inference


```
export OMP_NUM_THREADS=$(vCPUs/2)
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

# dummy data
python test.py --model=fcn --backbone=resnet101 --mode=val --eval --batch-size=1 --pretrained --quantized --benchmark

# real data
python test.py --model=fcn --backbone=resnet101 --mode=val --eval --batch-size=1 --pretrained --quantized

# deploy int8 model
python test.py --quantized --eval --deploy --model-prefix=./model/fcn_resnet101_voc-quantized-naive
```
Users also are recommended to bind processes to specific cores via `numactl` or `taskset` for better performance. 

## Performance

model | fp32 latency(ms) | s8 latency(ms) | fp32 pixAcc | fp32 mIoU | s8 pixAcc | s8 mIoU
-- | -- | -- | -- | -- | -- | -- |
fcn_resnet101_voc   |182.91 |37.97 |97.97% | 90.77 |98.00%  | 91.02 |
fcn_resnet101_coco  |192.65 |43.23 |91.28% | 62.40 |90.96%  | 61.73 |
psp_resnet101_voc   |252.17 | 94.06 | 98.46% | 93.29  | 98.45% | 93.26 |
psp_resnet101_coco  |253.49 | 94.32 | 91.82% | 69.48 | 91.88% | 69.92|
deeplab_resnet101_voc   |239.82 | 74.91 | 98.36% | 92.84 | 98.34% | 92.76 |
deeplab_resnet101_coco  |241.02 | 73.74 | 91.86% | 69.82 | 91.98% | 70.75 |

Please refer to [GluonCV Model Zoo](http://gluon-cv.mxnet.io/model_zoo/index.html#semantic-segmentation) for more avaliable pretrained models.

## References
1. Long, Jonathan, Evan Shelhamer, and Trevor Darrell. “Fully convolutional networks for semantic segmentation.” CVPR 2015.
