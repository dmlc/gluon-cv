# pylint: disable=all
"""Custom op: arange_like"""
import mxnet as mx
import numpy as np


class SliceLike(mx.operator.CustomOp):
    """Slice axis like the second input."""
    def __init__(self, axis, begin=0):
        self._axis = axis
        self._begin = begin

    def forward(self, is_train, req, in_data, out_data, aux):
        assert len(in_data) == 2
        assert len(out_data) == 1
        from_shape = in_data[1].shape
        self.assign(out_data[0], req[0], mx.nd.slice_axis(
            in_data[0], axis=self._axis, begin=self._begin,
            end=self._begin + from_shape[self._axis]))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        from_shape = in_data[1].shape
        self.assign(mx.nd.slice_axis(
            in_grad[0], axis=self._axis, begin=self._begin,
            end=self._begin + from_shape[self._axis]), req[0], out_grad[0])
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register("slice_like")
class SliceLikeProp(mx.operator.CustomOpProp):
    """Property of Slice axis like the second input."""
    def __init__(self, axis, begin=0):
        super(SliceLikeProp, self).__init__(True)
        self._axis = int(axis)
        self._begin = int(begin)

    def list_arguments(self):
        return ['data', 'data_shape']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shapes):
        assert len(in_shapes) == 2
        dshape = in_shapes[0]
        from_shape = in_shapes[1]
        if not self._axis < len(from_shape):
            raise IndexError(
                "Axis {} exceed input with shape {}".format(self._axis, str(from_shape)))
        if not dshape[self._axis] >= (self._begin + from_shape[self._axis]):
            raise IndexError(
                "Dimension {} of Axis {} exceed input with shape {}"
                .format(from_shape[self._axis] + self._begin, self._axis, dshape))
        outshape = [x for x in dshape]
        outshape[self._axis] = self._begin + from_shape[self._axis]
        return in_shapes, (outshape,), ()

    def create_operator(self, ctx, in_shapes, in_types):
        return SliceLike(self._axis, self._begin)


# class ArangeLike(mx.operator.CustomOp):
#     def __init__(self, axis):
#         self._axis = axis
#
#     def forward(self, is_train, req, in_data, out_data, aux):
#         assert len(in_data) == 1
#         assert len(out_data) == 1
#         in_shape = in_data[0].shape
#         if not self._axis < len(in_shape):
#             raise IndexError(
#                 "Axis {} exceed input with shape {}".format(self._axis, str(in_shape)))
#         self.assign(out_data[0], req[0], mx.nd.arange(in_shape[self._axis]))
#
#     def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
#         self.assign(in_grad[0], req[0], 0)
#
#
# @mx.operator.register("arange_like")
# class ArangeLikeProp(mx.operator.CustomOpProp):
#     def __init__(self, axis):
#         super(ArangeLikeProp, self).__init__(True)
#         self._axis = axis
#
#     def list_arguments(self):
#         return ['data']
#
#     def list_outputs(self):
#         return ['output']
#
#     def infer_shape(self, in_shapes):
#         dshape = in_shapes[0]
#         outshape = (dshape[self._axis],)
#         return (dshape,), (outshape,), ()
#
#     def create_operator(self, ctx, in_shapes, in_types):
#         return ArangeLike(self._axis)
