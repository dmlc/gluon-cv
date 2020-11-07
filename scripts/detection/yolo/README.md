# YOLO: You Only Look Once [1][2][3]

Currently V3 is implemented with training/evaluation/testing.

Random shape training is available through
```bash
python3 train_yolo3.py -h
```

Random shape training requires more GPU memory, but it provides better models. Alternatively, a normal fixed shape training is available as
```bash
python3 train_yolo3.py --no-random-shape
```


## Check out pre-trained model zoo
[GluonCV Model Zoo](http://gluon-cv.mxnet.io/model_zoo/index.html#object-detection)


## Inference/Calibration Tutorial

### Float32 Inference

```
python eval_yolo.py --network=mobilenet1.0 --gpus='' --dataset=voc --batch-size=1
```

### Calibration

Naive calibrate model by using 5 batch data (32 images per batch). Quantized model will be saved into `./model/`.

```
python eval_yolo.py --network=mobilenet1.0 --gpus='' --calibration --calib-mode=naive --dataset=voc
```

### INT8 Inference

```
python eval_yolo.py --network=mobilenet1.0 --gpus='' --deploy --model-prefix=./model/yolo3_mobilenet1.0_voc-quantized-naive  --dataset=voc --batch-size=1
```

## References
1. Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
2. Redmon, Joseph, and Ali Farhadi. "YOLO9000: better, faster, stronger." arXiv preprint (2017).
3. Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).
