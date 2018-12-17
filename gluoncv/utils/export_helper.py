"""Helper utils for export HybridBlock to symbols."""
from __future__ import absolute_import
import mxnet as mx
from mxnet.base import MXNetError
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn


class _DefaultPreprocess(HybridBlock):
    """Default preprocess block used by GluonCV.

    The default preprocess block includes:

        - mean [123.675, 116.28, 103.53]

        - std [58.395, 57.12, 57.375]

        - transpose to (B, 3, H, W)

    It is used to transform from resized original images with shape (1, H, W, 3) or (B, H, W, 3)
    in range (0, 255) and RGB color format.

    """
    def __init__(self, **kwargs):
        super(_DefaultPreprocess, self).__init__(**kwargs)
        with self.name_scope():
            mean = mx.nd.array([123.675, 116.28, 103.53]).reshape((1, 1, 1, 3))
            scale = mx.nd.array([58.395, 57.12, 57.375]).reshape((1, 1, 1, 3))
            self.init_mean = self.params.get_constant('init_mean', mean)
            self.init_scale = self.params.get_constant('init_scale', scale)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, init_mean, init_scale):
        x = F.broadcast_minus(x, init_mean)
        x = F.broadcast_div(x, init_scale)
        x = F.transpose(x, axes=(0, 3, 1, 2))
        return x

def export_block(path, block, data_shape=None, epoch=0, preprocess=True, layout='HWC',
                 ctx=mx.cpu()):
    """Helper function to export a HybridBlock to symbol JSON to be used by
    `SymbolBlock.imports`, `mxnet.mod.Module` or the C++ interface..

    Parameters
    ----------
    path : str
        Path to save model.
        Two files path-symbol.json and path-xxxx.params will be created,
        where xxxx is the 4 digits epoch number.
    block : mxnet.gluon.HybridBlock
        The hybridizable block. Note that normal gluon.Block is not supported.
    data_shape : tuple of int, default is None
        Fake data shape just for export purpose, in format (H, W, C).
        If you don't specify ``data_shape``, `export_block` will try use some common data_shapes,
        e.g., (224, 224, 3), (256, 256, 3), (299, 299, 3), (512, 512, 3)...
        If any of this ``data_shape`` goes through, the export will succeed.
    epoch : int
        Epoch number of saved model.
    preprocess : mxnet.gluon.HybridBlock, default is True.
        Preprocess block prior to the network.
        By default (True), it will subtract mean [123.675, 116.28, 103.53], divide
        std [58.395, 57.12, 57.375], and convert original image (B, H, W, C and range [0, 255]) to
        tensor (B, C, H, W) as network input. This is the default preprocess behavior of all GluonCV
        pre-trained models.
        You can use custom pre-process hybrid block or disable by set ``preprocess=None``.
    layout : str, default is 'HWC'
        The layout for raw input data. By default is HWC. Supports 'HWC' and 'CHW'.
        Note that image channel order is always RGB.
    ctx: mx.Context, default mx.cpu()
        Network context.

    Returns
    -------
    None

    """
    # input image layout
    if data_shape is None:
        data_shapes = [(s, s, 3) for s in (224, 256, 299, 300, 320, 416, 512, 600)]
    else:
        data_shapes = [data_shape]

    if preprocess:
        # add preprocess block
        if preprocess is True:
            preprocess = _DefaultPreprocess()
        else:
            if not isinstance(preprocess, HybridBlock):
                raise TypeError("preprocess must be HybridBlock, given {}".format(type(preprocess)))
        wrapper_block = nn.HybridSequential()
        preprocess.initialize(ctx=ctx)
        wrapper_block.add(preprocess)
        wrapper_block.add(block)
    else:
        wrapper_block = block
    wrapper_block.collect_params().reset_ctx(ctx)

    # try different data_shape if possible, until one fits the network
    last_exception = None
    for dshape in data_shapes:
        h, w, c = dshape
        if layout == 'HWC':
            x = mx.nd.zeros((1, h, w, c), ctx=ctx)
        elif layout == 'CHW':
            x = mx.nd.zeros((1, c, h, w), ctx=ctx)

        # hybridize and forward once
        wrapper_block.hybridize()
        try:
            wrapper_block(x)
            wrapper_block.export(path, epoch)
            last_exception = None
            break
        except MXNetError as e:
            last_exception = e
    if last_exception is not None:
        raise RuntimeError(str(last_exception).splitlines()[0])
