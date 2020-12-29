# Prepare large datasets for vision
[Gluon](https://mxnet.incubator.apache.org/gluon/) itself provides self-managed
tiny datasets such as MNIST, CIFAR-10/100, Fashion-MNIST.
However, downloading and unzipping large scale datasets are very time consuming
processes which are not appropriate to be initialized during class instantiation.
Therefore we provide convenient example scripts for existing/non-existing datasets.

All datasets requires one-time setup, and will be automatically recognized by `gluoncv`
package in the future.

## Instructions
Please refer to our official [tutorials](http://gluon-cv.mxnet.io/build/examples_datasets/index.html)
