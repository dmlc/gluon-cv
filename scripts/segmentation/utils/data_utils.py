import importlib
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

__all__ = ['get_data_loader']

def get_data_loader(args):
    dataset = importlib.import_module('gluonvision.data.' + \
        args.dataset + '.segmentation')
    return _getDataloader(args, dataset)

def _getDataloader(args, dataset):
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.mean, args.std)
    ])

    if args.test:
        testset = dataset._Segmentation(
            split='test',
            transform=input_transform)
        test_data = gluon.data.DataLoader(
            testset, args.test_batch_size,
            last_batch='keep',
            batchify_fn=_test_batchify_fn,
            num_workers=args.workers)
        return test_data

    trainset = dataset._Segmentation(
        split='train',
        transform=input_transform)
    testset = dataset._Segmentation(
        split='val',
        transform=input_transform)

    train_data = gluon.data.DataLoader(
        trainset, args.batch_size, shuffle=True, last_batch='rollover',
        num_workers=args.workers)
    test_data = gluon.data.DataLoader(testset, args.test_batch_size,
        last_batch='keep', num_workers=args.workers)
    return train_data, test_data


def _test_batchify_fn(self, data):
    if isinstance(data[0], (str, mx.nd.NDArray)):
        return list(data)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [self._batchify_fn(i) for i in data]
    else:
        data = np.asarray(data)
        return mx.nd.array(data, dtype=data.dtype)
