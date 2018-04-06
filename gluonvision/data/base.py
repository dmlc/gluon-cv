"""Base dataset methods."""
import os
from mxnet.gluon.data import dataset


class VisionDataset(dataset.Dataset):
    """Base Dataset with directory checker.

    Parameters
    ----------
    root : str
        The root path of xxx.names, by defaut is '~/.mxnet/datasets/foo', where
        `foo` is the name of the dataset.
    """
    def __init__(self, root):
        if not os.path.isdir(os.path.expanduser(root)):
            helper_msg = "{} is not a valid dir. Did you forget to initalize \
                         datasets described in `repo/scripts/datasets`? You need \
                         to initialize each dataset only once.".format(root)
            raise OSError(helper_msg)

    @property
    def classes(self):
        raise NotImplementedError

    @property
    def class_names(self):
        raise NotImplementedError
