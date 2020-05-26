# pylint: disable=arguments-differ,line-too-long,missing-docstring,missing-module-docstring
from functools import partial
import mxnet as mx
from mxnet.gluon.nn import HybridBlock

__all__ = ['DropBlock', 'set_drop_prob', 'DropBlockScheduler']

class DropBlock(HybridBlock):
    def __init__(self, drop_prob, block_size, c, h, w):
        super(DropBlock, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.c, self.h, self.w = c, h, w
        self.numel = c * h * w
        pad_h = max((block_size - 1), 0)
        pad_w = max((block_size - 1), 0)
        self.padding = (pad_h//2, pad_h-pad_h//2, pad_w//2, pad_w-pad_w//2)
        self.dtype = 'float32'

    def hybrid_forward(self, F, x):
        if not mx.autograd.is_training() or self.drop_prob <= 0:
            return x
        gamma = self.drop_prob * (self.h * self.w) / (self.block_size ** 2) / \
            ((self.w - self.block_size + 1) * (self.h - self.block_size + 1))
        # generate mask
        mask = F.random.uniform(0, 1, shape=(1, self.c, self.h, self.w), dtype=self.dtype) < gamma
        mask = F.Pooling(mask, pool_type='max',
                         kernel=(self.block_size, self.block_size), pad=self.padding)
        mask = 1 - mask
        y = F.broadcast_mul(F.broadcast_mul(x, mask),
                            (1.0 * self.numel / mask.sum(axis=0, exclude=True).expand_dims(1).expand_dims(1).expand_dims(1)))
        return y

    def cast(self, dtype):
        super(DropBlock, self).cast(dtype)
        self.dtype = dtype

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' + \
            'drop_prob: {}, block_size{}'.format(self.drop_prob, self.block_size) +')'
        return reprstr

def set_drop_prob(drop_prob, module):
    """
    Example:
        from functools import partial
        apply_drop_prob = partial(set_drop_prob, 0.1)
        net.apply(apply_drop_prob)
    """
    if isinstance(module, DropBlock):
        module.drop_prob = drop_prob


class DropBlockScheduler(object):
    # pylint: disable=chained-comparison
    def __init__(self, net, start_prob, end_prob, num_epochs):
        self.net = net
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.num_epochs = num_epochs

    def __call__(self, epoch):
        ratio = self.start_prob + 1.0 * (self.end_prob - self.start_prob) * (epoch + 1) / self.num_epochs
        assert (ratio >= 0 and ratio <= 1)
        apply_drop_prob = partial(set_drop_prob, ratio)
        self.net.apply(apply_drop_prob)
        self.net.hybridize()
