from __future__ import print_function
from __future__ import division

import mxnet as mx
import numpy as np

import gluoncv as gcv
from gluoncv.data.batchify import *
from gluoncv.data import DetectionDataLoader


class DummyDetectionDataset(object):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        num_object = np.random.randint(1, 6)
        return mx.random.normal(shape=(300, 300, 3)), np.random.randn(num_object, 5)

def test_detection_dataloader():
    dataset = DummyDetectionDataset(8)
    for num_workers in [0, 1, 2, 4]:
        for shuffle in (True, False):
            for last_batch in ('keep', 'discard', 'rollover'):
                dataloader = DetectionDataLoader(
                    dataset, batch_size=2, shuffle=shuffle, last_batch=last_batch,
                    num_workers=num_workers)
                for batch in dataloader:
                    mx.nd.waitall()
                    pass

                # new dataloader methods
                batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
                dataloader = mx.gluon.data.DataLoader(
                    dataset, batch_size=2, shuffle=shuffle, last_batch=last_batch,
                    batchify_fn=batchify_fn, num_workers=num_workers)
                for batch in dataloader:
                    mx.nd.waitall()
                    pass

                batchify_fn = Tuple(Append(), Append())
                dataloader = mx.gluon.data.DataLoader(
                    dataset, batch_size=2, shuffle=shuffle, last_batch=last_batch,
                    batchify_fn=batchify_fn, num_workers=num_workers)
                for batch in dataloader:
                    mx.nd.waitall()
                    pass

if __name__ == '__main__':
    import nose
    nose.runmodule()
