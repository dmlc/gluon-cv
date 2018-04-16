# Single Shot Multibox Object Detection [1]

## Performance
PASCAL VOC 2007 Test Mean Average Precision (mAP)

| Model | Base size | Original | GluonVision * |
|:-:|:-:|:-:|:-:|
| VGG16 Atrous | 300 | 77.5 % [2] | 77.9 % |
| VGG16 Atrous | 512 | 79.5 % [2] | 79.6 % |
| ResNet50_v1  | 512 | - | 80.1 |

\* Single time training results may vary because of randomness.
\* You are very welcome to contribute models with better performances to the community.


## References
1. Wei Liu, et al. "SSD: Single shot multibox detector" ECCV 2016.
2. Cheng-Yang Fu, et al. "[DSSD : Deconvolutional Single Shot Detector](https://arxiv.org/abs/1701.06659)" arXiv 2017.
