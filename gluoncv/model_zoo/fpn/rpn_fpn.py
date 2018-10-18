"""Region Proposal Networks Head Definition."""
from __future__ import absolute_import

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

class RPNFPNHead(gluon.HybridBlock):
    r"""Region Proposal Network Head.

    Parameters
    ----------
    channels : int
        Channel number used in convolutional layers.
    anchor_depth : int
        Each FPN stage have one scale and three ratios,
        we can compute anchor_depth = len(scale) \times len(ratios)
    """
    def __init__(self, channels, anchor_depth, **kwargs):
        super(RPNFPNHead, self).__init__(**kwargs)
        weight_initializer = mx.init.Normal(0.01)
        with self.name_scope():
            self.conv1 = nn.HybridSequential()
            self.conv1.add(nn.Conv2D(channels, 3, 1, 1, weight_initializer=weight_initializer))
            self.conv1.add(nn.Activation('relu'))
            # use sigmoid instead of softmax, reduce channel numbers
            # Note : that is to say, if use softmax here, then the self.score will anchor_depth*2 output channel
            self.score = nn.Conv2D(anchor_depth, 1, 1, 0, weight_initializer=weight_initializer)
            self.loc = nn.Conv2D(anchor_depth * 4, 1, 1, 0, weight_initializer=weight_initializer)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        """Forward RPN Head.

        This HybridBlock will generate predicted values for cls and box.

        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
            Feature tensor. With (1, C, H, W) shape

        Returns
        -------
        (rpn_scores, rpn_boxes, raw_rpn_scores, raw_rpn_boxes)
            Returns predicted scores and regions.

        """
        '''3x3 conv with relu activation'''
        x = self.conv1(x)
        '''(1, C, H, W)->(1, 9, H, W)->(1, H, W, 9)->(1, H*W*9, 1)'''
        raw_rpn_scores = self.score(x).transpose(axes=(0, 2, 3, 1)).reshape((0, -1, 1))
        '''(1, H*W*9, 1)'''
        rpn_scores = F.sigmoid(F.stop_gradient(raw_rpn_scores))
        '''(1, C, H, W)->(1, 36, H, W)->(1, H, W, 36)->(1, H*W*9, 4)'''
        raw_rpn_boxes = self.loc(x).transpose(axes=(0, 2, 3, 1)).reshape((0, -1, 4))
        '''(1, H*W*9, 1)'''
        rpn_boxes = F.stop_gradient(raw_rpn_boxes)
        # return raw predictions as well in training for bp
        return rpn_scores, rpn_boxes, raw_rpn_scores, raw_rpn_boxes