import mxnet as mx
import numpy as np


class BBoxClipToImage(mx.operator.CustomOp):
    """Clip bounding box to image edges.

    Parameters
    ----------
    axis : int
        The coordinate axis with length 4.

    """
    def __init__(self, axis=-1):
        self.axis = int(axis)

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        shape_like = in_data[1]
        height, width = shape_like.shape[-2:]
        assert x.shape[self.axis] == 4
        xmin, ymin, xmax, ymax = x.split(axis=self.axis, num_outputs=4)
        xmin = xmin.clip(0, width - 1)
        ymin = ymin.clip(0, height - 1)
        xmax = xmax.clip(0, width - 1)
        ymax = ymax.clip(0, height - 1)
        out = mx.nd.concat(xmin, ymin, xmax, ymax, dim=self.axis)
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register('bbox_clip_to_image')
class BBoxClipToImageProp(mx.operator.CustomOpProp):
    """Property of BBoxClipToImage custom Op.

    Parameters
    ----------
    axis : int
        The coordinate axis with length 4.

    """
    def __init__(self, axis=-1):
        super(BBoxClipToImageProp, self).__init__(need_top_grad=True)
        self.axis = int(axis)

    def list_arguments(self):
        return ['data', 'shape_like']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []

    def infer_type(self, in_type):
        return [in_type[0], in_type[0]], [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return BBoxClipToImage(self.axis)
