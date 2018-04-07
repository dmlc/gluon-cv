import importlib
import numpy as np
import mxnet as mx
from mxnet import gluon

__all__ = ['get_data_loader', 'Compose', 'ToTensor', 'ToLabel', 'Normalize']

def get_data_loader(args):
    dataset = importlib.import_module('gluonvision.data.' + \
        args.dataset + '.segmentation')
    return _getDataloader(args, dataset)

def _getDataloader(args, dataset):
    ctx = mx.cpu(0)
    input_transform = Compose([
        ToTensor(ctx=ctx),
        Normalize(args.mean, args.std, ctx)])
    target_transform = Compose([
        ToLabel(ctx=ctx)])

    if args.test:
        testset = dataset._Segmentation(
            root=args.data_folder,
            split='test',
            transform=input_transform)
        return testset

    trainset = dataset._Segmentation(
        root=args.data_folder,
        split='train',
        transform=input_transform,
        target_transform=target_transform)
    testset = dataset._Segmentation(
        root=args.data_folder,
        split='val',
        transform=input_transform,
        target_transform=target_transform)

    train_data = gluon.data.DataLoader(
        trainset, args.batch_size, shuffle=True, last_batch='discard',
        num_workers=args.workers)
    test_data = gluon.data.DataLoader(testset, args.test_batch_size,
        last_batch='discard', num_workers=args.workers)
    return train_data, test_data


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor(object):
    """Convert PIL image to NdArray
    """
    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, img):
        img = mx.nd.array(np.array(img).transpose(2, 0, 1).astype('float32'), 
            ctx=self.ctx) / 255
        return img


class ToLabel(object):
    """Convert PIL image to labels
    """
    def __init__(self, ctx):
        self.ctx = ctx
    def __call__(self, label):
        return mx.nd.array(np.array(label), ctx=self.ctx).astype('int32')


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, ctx):
        self.mean = mx.nd.array(mean, ctx=ctx)
        self.std = mx.nd.array(std, ctx=ctx)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return (tensor - self.mean.reshape(shape=(3,1,1))) / self.std .reshape(shape=(3,1,1))
