# Pose Estimation[1]
[GluonCV Model Zoo](http://gluon-cv.mxnet.io/model_zoo/index.html#semantic-segmentation)

## Inference/Calibration Tutorial

### FP32 inference

```

export OMP_NUM_THREADS=$(vCPUs/2)
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

# dummy data
python validate.py --model simple_pose_resnet50_v1b --num-joints 17 --batch-size 32 --benchmark

# real data
python validate.py --model simple_pose_resnet50_v1b --num-joints 17 --batch-size 32
```

### Calibration

Naive calibrated model by using 5 batch data (32 images per batch). Quantized model will be saved into `./model/`.

```
# coco keypoint dataset
python validate.py --model simple_pose_resnet50_v1b --batch-size=1 --num-joints 17 --calibration
```

### INT8 Inference

```

export OMP_NUM_THREADS=$(vCPUs/2)
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

# dummy data
python validate.py --model simple_pose_resnet50_v1d --num-joints 17 --quantized --benchmark

# real data
python validate.py --model simple_pose_resnet50_v1d --num-joints 17 --quantized

# deploy static model
python validate.py --deploy --model-prefix ./model/simple_pose_resnet50_v1d-quantized-naive --num-joints 17 --model simple_pose

```

Users are also recommended to bind processes to specific cores via `numactl` for better performance, like below:

```
numactl --physcpubind=0-27 --membind=0 python validate.py ...
```

## Performance

model | fp32 latency(ms) | s8 latency(ms) | fp32 pixAcc | fp32 mIoU | s8 pixAcc | s8 mIoU
-- | -- | -- | -- | -- | -- | -- |
simple_pose_resnet18_v1b    | | |97.97% | 90.77 |98.00%  | 91.02 |
simple_pose_resnet50_v1b    | | |91.28% | 62.40 |90.96%  | 61.73 |
simple_pose_resnet50_v1d    | |  | 98.46% | 93.29  | 98.45% | 93.26 |
simple_pose_resnet101_v1b   | | 94.32 | 91.82% | 69.48 | 91.88% | 69.92|
simple_pose_resnet101_v1d   | | 74.91 | 98.36% | 92.84 | 98.34% | 92.76 |



## References

