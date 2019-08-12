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
Users also are recommended to bind processes to specific cores via `numactal` or `taskset` for better performance. 

## Performance

model | fp32 latency(ms) | s8 latency(ms) | fp32 pixAcc | s8 pixAcc
-- | -- | -- | -- | -- |
fcn_resnet101_voc   |182.91 |37.97 |97.97% |96.53%  |
fcn_resnet101_coco  |192.65 |43.23 |91.28% |90.96%  |

Please refer to [GluonCV Model Zoo](http://gluon-cv.mxnet.io/model_zoo/index.html#semantic-segmentation) for more avaliable pretrained models.

## References
1. Long, Jonathan, Evan Shelhamer, and Trevor Darrell. “Fully convolutional networks for semantic segmentation.” CVPR 2015.
