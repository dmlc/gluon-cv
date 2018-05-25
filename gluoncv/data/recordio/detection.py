"""Detection dataset from RecordIO files."""
from __future__ import absolute_import
from __future__ import division
import numpy as np
from mxnet import gluon


class RecordFileDetection(gluon.data.vision.ImageRecordDataset):
    """Detection dataset loaded from record file.
    The supported record file is using the same format used by
    :meth:`mxnet.image.ImageDetIter` and :meth:`mxnet.io.ImageDetRecordIter`.

    .. note::

        We suggest you to use ``RecordFileDetection`` only if you are familier with
        the record files.

    Parameters
    ----------
    filename : str
        Path of the record file. It require both *.rec and *.idx file in the same
        directory, where raw image and labels are stored in *.rec file for better
        IO performance, *.idx file is used to provide random access to the binary file.

    Examples
    --------
    >>> record_dataset = RecordFileDetection('train.rec')
    >>> img, label = record_dataset[0]
    >>> print(img.shape, label.shape)
    (512, 512, 3) (1, 5)

    """
    def __init__(self, filename):
        super(RecordFileDetection, self).__init__(filename)

    def __getitem__(self, idx):
        img, label = super(RecordFileDetection, self).__getitem__(idx)
        h, w, _ = img.shape
        label = self._transform_label(label, h, w)
        return img, label

    def _transform_label(self, label, height, width):
        label = np.array(label).ravel()
        header_len = int(label[0])  # label header
        label_width = int(label[1])  # the label width for each object, >= 5
        if label_width < 5:
            raise ValueError(
                "Label info for each object shoudl >= 5, given {}".format(label_width))
        min_len = header_len + 5
        if len(label) < min_len:
            raise ValueError(
                "Expected label length >= {}, got {}".format(min_len, len(label)))
        if (len(label) - header_len) % label_width:
            raise ValueError(
                "Broken label of size {}, cannot reshape into (N, {}) "
                "if header length {} is excluded".format(len(label), label_width, header_len))
        gcv_label = label[header_len:].reshape(-1, label_width)
        # swap columns, gluon-cv requires [xmin-ymin-xmax-ymax-id-extra0-extra1-xxx]
        ids = gcv_label[:, 0].copy()
        gcv_label[:, :4] = gcv_label[:, 1:5]
        gcv_label[:, 4] = ids
        # restore to absolute coordinates
        gcv_label[:, (0, 2)] *= width
        gcv_label[:, (1, 3)] *= height
        return gcv_label
