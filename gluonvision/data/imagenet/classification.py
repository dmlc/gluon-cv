"""ImageNet classification dataset."""
import os
from mxnet.gluon.data.vision import ImageFolderDataset

__all__ = ['ImageNet']

class ImageNet(ImageFolderDataset):
    """ImageNet classification dataset from http://www.image-net.org/.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/imagenet'
        Path to temp folder for storing data.
    train : bool, default True
        Whether to load the training or testing set.
    transform : function, default None
        A user defined callback that transforms each sample. For example:
    ::
        transform=lambda data, label: (data.astype(np.float32)/255, label)

    """
    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'imagenet'),
                 train=True, transform=None):
        split = 'train' if train else 'val'
        root = os.path.join(root, split)
        super(ImageNet, self).__init__(root=root, flag=1, transform=transform)
