# Prepare large datasets for vision
[Gluon](https://mxnet.incubator.apache.org/gluon/) itself provides self-managed
tiny datasets such as MNIST, CIFAR-10/100, Fashion-MNIST.
However, downloading and unzipping large scale datasets are very time consuming
processes which are not appropriate to be initialized during class instantiation.
Therefore we provide convenient example scripts for existing/non-existing datasets.

All datasets requires one-time setup, and will be automatically recognized by `gluonvision`
package in the future.

## Examples

- Create symbolic link for existing VOCdevkit

```
python examples/setup_pascal_voc.py ~/datasets/VOCdevkt
```
- Download PASCAL VOC dataset to `~/datasets/VOCdevkit` and make symlink

```
python example/setup_pascal_voc.py ~/datasets/VOCdevkit --download
```

- if you insist, overwrite downloaded files if exist

```
python example/setup_pascal_voc.py ~/datasets/VOCdevkit --download --overwrite
```
