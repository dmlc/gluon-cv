"""Region Proposal Networks Definition."""
from __future__ import absolute_import

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn
from .anchor import RPNAnchorGenerator
from .proposal import RPNProposal


class RPN(gluon.HybridBlock):
    r"""Region Proposal Network.

    Parameters
    ----------
    channels : int
        Channel number used in convolutional layers.
    stride : int
        Feature map stride with respect to original image.
        This is usually the ratio between original image size and feature map size.
    base_size : int
        The width(and height) of reference anchor box.
    scales : iterable of float
        The areas of anchor boxes.
        We use the following form to compute the shapes of anchors:

        .. math::

            width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
            height_{anchor} = size_{base} \times scale \times \sqrt{ratio}

    ratios : iterable of float
        The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.
    clip : float
        Clip bounding box target to this value.
    nms_thresh : float
        IOU threshold for NMS. It is used to remove overlapping proposals.
    train_pre_nms : int
        Filter top proposals before NMS in training.
    train_post_nms : int
        Return top proposal results after NMS in training.
    test_pre_nms : int
        Filter top proposals before NMS in testing.
    test_post_nms : int
        Return top proposal results after NMS in testing.
    min_size : int
        Proposals whose size is smaller than ``min_size`` will be discarded.

    """
    def __init__(self, channels, stride, base_size, scales, ratios, alloc_size,
                 clip, nms_thresh, train_pre_nms, train_post_nms,
                 test_pre_nms, test_post_nms, min_size, **kwargs):
        super(RPN, self).__init__(**kwargs)
        weight_initializer = mx.init.Normal(0.01)
        with self.name_scope():
            self.anchor_generator = RPNAnchorGenerator(
                stride, base_size, ratios, scales, alloc_size)
            anchor_depth = self.anchor_generator.num_depth
            self.region_proposaler = RPNProposal(
                clip, nms_thresh, train_pre_nms, train_post_nms,
                test_pre_nms, test_post_nms, min_size, stds=(1., 1., 1., 1.))
            self.conv1 = nn.HybridSequential()
            self.conv1.add(nn.Conv2D(channels, 3, 1, 1, weight_initializer=weight_initializer))
            self.conv1.add(nn.Activation('relu'))
            # use sigmoid instead of softmax, reduce channel numbers
            self.score = nn.Conv2D(anchor_depth, 1, 1, 0, weight_initializer=weight_initializer)
            self.loc = nn.Conv2D(anchor_depth * 4, 1, 1, 0, weight_initializer=weight_initializer)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, img):
        """Forward RPN.

        The behavior during traing and inference is different.

        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
            Feature tensor.
        img : mxnet.nd.NDArray or mxnet.symbol
            The original input image.

        Returns
        -------
        (rpn_score, rpn_box)
            Returns predicted scores and regions which are candidates of objects.

        """
        anchors = self.anchor_generator(x)
        x = self.conv1(x)
        raw_rpn_scores = self.score(x).transpose(axes=(0, 2, 3, 1)).reshape((0, -1, 1))
        rpn_scores = F.sigmoid(F.stop_gradient(raw_rpn_scores))
        rpn_box_pred = self.loc(x).transpose(axes=(0, 2, 3, 1)).reshape((0, -1, 4))
        rpn_score, rpn_box = self.region_proposaler(
            anchors, rpn_scores, F.stop_gradient(rpn_box_pred), img)
        if autograd.is_training():
            # return raw predictions as well in training for bp
            return rpn_score, rpn_box, raw_rpn_scores, rpn_box_pred, anchors
        return rpn_score, rpn_box
