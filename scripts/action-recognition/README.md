# Action Recognition[1]
[GluonCV Model Zoo](https://gluon-cv.mxnet.io/model_zoo/action_recognition.html)

## Inference/Calibration Tutorial

### FP32 inference

```

export CPUs=`lscpu | grep 'Core(s) per socket' | awk '{print $4}'`
export OMP_NUM_THREADS=${CPUs}
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

# dummy data
python test_recognizer.py --model inceptionv3_ucf101 --use-pretrained --use-tsn --mode hybrid --input-size 299 --new-height 340 --new-width 450 --num-segments 3 --batch-size 64 --benchmark

# real data
python test_recognizer.py --model inceptionv3_ucf101 --use-pretrained --use-tsn --mode hybrid --input-size 299 --new-height 340 --new-width 450 --num-segments 3 --batch-size 64
```

### Calibration

Naive calibrated model by using 5 batch data (32 images per batch). Quantized model will be saved into `./model/`.

```
# ucf101 dataset
python test_recognizer.py --model inceptionv3_ucf101 --new-height 340 --new-width 450 --input-size 299 --num-segments 3 --use-pretrained --use-tsn --calibration
```

### INT8 Inference

```

export CPUs=`lscpu | grep 'Core(s) per socket' | awk '{print $4}'`
export OMP_NUM_THREADS=${CPUs}
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

# dummy data
python test_recognizer.py --model inceptionv3_ucf101 --mode hybrid --input-size 299 --new-height 340 --new-width 450 --batch-size 64 --num-segments 3 --quantized --benchmark

# real data
python test_recognizer.py --model inceptionv3_ucf101 --mode hybrid --input-size 299 --new-height 340 --new-width 450 --batch-size 64 --num-segments 3 --quantized

# deploy static model
python test_recognizer.py --model ucf --deploy --model-prefix ./model/inceptionv3_ucf101-quantized-naive --input-size 299 --new-height 340 --new-width 450 --batch-size 64 --num-segments 3 --benchmark

```

Users are also recommended to bind processes to specific cores via `numactl` for better performance, like below:

```
numactl --physcpubind=0-27 --membind=0 python test_recognizer.py ...
```

## Performance
Below results are collected based on Intel(R) VNNI enabled C5.12xlarge with 24 physical cores.

|model | fp32 latency(ms) | s8 latency(ms) | fp32 Top-1 | s8 Top-1 |
|-- | -- | -- | -- | -- |
inceptionv3_ucf101    |13.02  |4.70 |86.92 | 86.55 |
vgg16_ucf101          |15.68  |4.58 |81.86 | 81.41 |

## References

1. Limin Wang, Yuanjun Xiong, Zhe Wang and Yu Qiao. “Towards Good Practices for Very Deep Two-Stream ConvNets.” arXiv preprint arXiv:1507.02159, 2015.
