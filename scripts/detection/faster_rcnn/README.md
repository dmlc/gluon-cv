# Faster R-CNN: Towards real-time object detection with region proposal networks. [1]

[GluonCV Model Zoo](http://gluon-cv.mxnet.io/model_zoo/index.html#object-detection)

- `--amp` Use [Automatic Mixed Precision training](https://mxnet.incubator.apache.org/versions/master/tutorials/amp/amp_tutorial.html), automatically casting FP16 where safe.
- `--horovod` Use [Horovod](https://github.com/horovod/horovod) for distributed training, with a network agnostic wrapper for the optimizer, allowing efficient allreduce using OpemMPI and NCCL.

## References
1. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. "Faster R-CNN: Towards real-time object detection with region proposal networks." In IEEE Transactions on Pattern Analysis and Machine Intelligence, 2016.
2. Ross Girshick. "Fast R-CNN." In Proceedings of the IEEE International Conference on Computer Vision, 2015.
