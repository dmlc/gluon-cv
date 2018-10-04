"""Market 1501 Person Re-Identification Dataset."""
from mxnet.gluon.data import dataset
from mxnet import image

__all__ = ['ImageTxtDataset']


class ImageTxtDataset(dataset.Dataset):
    """Load the Market 1501 dataset.

    Parameters
    ----------
    items : list
        List for image names and labels.
    flag : int, default 1
        Whether to load the color image or gray image.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self, items, flag=1, transform=None):
        self._flag = flag
        self._transform = transform
        self.items = items

    def __getitem__(self, idx):
        fpath = self.items[idx][0]
        img = image.imread(fpath, self._flag)
        label = self.items[idx][1]
        if self._transform is not None:
            img = self._transform(img)
        return img, label

    def __len__(self):
        return len(self.items)
