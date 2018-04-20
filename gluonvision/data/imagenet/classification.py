"""ImageNet classification dataset."""
from os import path
from mxnet.gluon.data.vision import ImageFolderDataset

__all__ = ['ImageNet']

class ImageNet(ImageFolderDataset):
    """Load the ImageNet classification dataset.

    Refer to :doc:`../build/examples_datasets/imagenet` for the description of
    this dataset and how to prepare it.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/imagenet'
        Path to the folder stored the dataset.
    train : bool, default True
        Whether to load the training or validation set.
    transform : function, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples. (TODO, should we restrict its datatype
        to transformer?)
    """
    def __init__(self, root=path.join('~', '.mxnet', 'datasets', 'imagenet'),
                 train=True, transform=None):
        split = 'train' if train else 'val'
        root = path.join(root, split)
        super(ImageNet, self).__init__(root=root, flag=1, transform=transform)
