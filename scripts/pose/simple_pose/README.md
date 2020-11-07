# Pose Estimation[1]
[GluonCV Model Zoo](http://gluon-cv.mxnet.io/model_zoo/index.html#pose-estimation)

## Inference/Calibration Tutorial

### FP32 inference

```

export CPUs=`lscpu | grep 'Core(s) per socket' | awk '{print $4}'`
export OMP_NUM_THREADS=${CPUs}
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

export CPUs=`lscpu | grep 'Core(s) per socket' | awk '{print $4}'`
export OMP_NUM_THREADS=${CPUs}
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
Below results are collected based on Intel(R) VNNI enabled C5.12xlarge with 24 physical cores.

|model | fp32 latency(ms) | s8 latency(ms) | fp32 OKS AP | s8 OKS AP |
|-- | -- | -- | -- | -- |
simple_pose_resnet18_v1b    |5.55  |2.38 |66.3 | 65.9 |
simple_pose_resnet50_v1b    |12.68 |4.65 |71.0 | 70.6 |
simple_pose_resnet50_v1d    |18.91 |4.80 |71.6 | 71.4 |
simple_pose_resnet101_v1b   |21.84 |6.97 |72.4 | 72.2 |
simple_pose_resnet101_v1d   |27.15 |7.15 |73.0 | 72.7 |

## References

1. Xiao, Bin, Haiping Wu, and Yichen Wei. “Simple baselines for human pose estimation and tracking.” Proceedings of the European Conference on Computer Vision (ECCV). 2018
