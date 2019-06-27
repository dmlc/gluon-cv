# Single Shot Multibox Object Detection [1]

[GluonCV Model Zoo](http://gluon-cv.mxnet.io/model_zoo/index.html#object-detection)

- `--dali` Use [DALI](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html) for faster data loading and data preprocessing in training with COCO dataset. DALI >= 0.12 required.
- `--amp` Use [Automatic Mixed Precision training](https://mxnet.incubator.apache.org/versions/master/tutorials/amp/amp_tutorial.html), automatically casting FP16 where safe.
- `--horovod` Use [Horovod](https://github.com/horovod/horovod) for distributed training, with a network agnostic wrapper for the optimizer, allowing efficient allreduce using OpemMPI and NCCL.

## References
1. Wei Liu, et al. "SSD: Single shot multibox detector" ECCV 2016.
2. Cheng-Yang Fu, et al. "[DSSD : Deconvolutional Single Shot Detector](https://arxiv.org/abs/1701.06659)" arXiv 2017.
