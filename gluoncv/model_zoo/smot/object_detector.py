"""
MXNet implementation of SMOT: Single-Shot Multi Object Tracking
https://arxiv.org/abs/2010.16031
"""
import os
import numpy as np
import mxnet as mx
from mxnet.gluon.contrib.nn import SyncBatchNorm

from ..model_store import get_model_file
from .ssd_bifpn import get_ssd as get_bifpn_ssd
from .utils import mxnet_frame_preprocessing, timeit_context
from .utils import remap_bboxes as _remap_bboxes


def get_bifpn_net(classes,
                  data_shape,
                  pretrain_weight_file=None,
                  use_keypoints=True,
                  return_features=False,
                  root=os.path.join('~', '.mxnet', 'models'),
                  ctx=None,
                  **kwargs):
    """
    Load bifpn_ssd model with mobilenet backbone
    """

    anchor_sizes = [[.02 * data_shape],
                    [.04 * data_shape, .06 * data_shape],
                    [.08 * data_shape, .12 * data_shape, .16 * data_shape],
                    [.2 * data_shape, .25 * data_shape, .3 * data_shape],
                    [.4 * data_shape, .5 * data_shape],
                    [.6 * data_shape, .75 * data_shape, .9 * data_shape]]

    anchor_ratios = [[1, 1 / 3], [1, 1 / 3, ], [1, 1 / 3, ], [1, 1 / 3, 1.5], [1, 1 / 3, 1.5], [1, 1 / 3, 1.5]]
    pretrained_base = False
    features = ['relu6_fwd', 'relu10_fwd', 'relu22_fwd', 'relu25_fwd']
    ssd_filters = [128, 128]
    fpn_filters = 64
    steps = [4, 8, 16, 32, 64, 128]

    net = get_bifpn_ssd("MobileNet1.0",
                        data_shape,
                        features=features,
                        ssd_filters=ssd_filters,
                        sizes=anchor_sizes,
                        ratios=anchor_ratios,
                        steps=steps,
                        classes=classes,
                        dataset='',
                        pretrained_base=pretrained_base,
                        is_fpn=True,
                        fpn_filters=fpn_filters,
                        is_multitask=False,
                        use_keypoints=use_keypoints,
                        return_features=return_features,
                        **kwargs)

    if pretrain_weight_file is None:
        net.load_parameters(get_model_file('smot_ssd_bifpn_mobilenet', tag=True, root=root), ctx=ctx)
    else:
        net.load_parameters(pretrain_weight_file, ctx=ctx)

    return net


class JointObjectDetector:
    """
    Define the general single shot object detector
    """
    def __init__(self, gpu_id, data_shape,
                 aspect_ratio=16/9., network_param=None):
        self.ctx = mx.gpu(gpu_id)
        self.data_shape = data_shape
        self.net = get_bifpn_net(classes=['face', 'person'],
                                 data_shape=self.data_shape,
                                 pretrain_weight_file=network_param,
                                 ctx=self.ctx,
                                 return_features=False,
                                 use_keypoints=False,
                                 norm_layer=SyncBatchNorm,
                                 norm_kwargs={'num_devices': 1},
                                 )

        self.anchor_tensor = None
        self._anchor_image_shape = (1, 1)
        self._anchor_num = 1

        self.mean_mx = mx.nd.array(np.array([0.485, 0.456, 0.406])).as_in_context(self.ctx)
        self.std_mx = mx.nd.array(np.array([0.229, 0.224, 0.225])).as_in_context(self.ctx)
        self.ratio = aspect_ratio

    def run_detection(self, image, tracking_box_indices, tracking_box_weights, tracking_box_classes):
        """

        Parameters
        ----------
        image: RGB images

        Returns
        -------

        """

        with timeit_context("preprocess"):
            data_tensor, padded_w, padded_h, expand = mxnet_frame_preprocessing(image,
                                                                                self.data_shape,
                                                                                self.ratio,
                                                                                self.mean_mx,
                                                                                self.std_mx,
                                                                                self.ctx)
            mx.nd.waitall()


        with timeit_context("network"):
            real_tracking_indices = tracking_box_indices + tracking_box_classes * self._anchor_num
            ids, scores, detection_bboxes, detection_anchor_indices, tracking_results, anchors = self.net(
                data_tensor.as_in_context(self.ctx), None, real_tracking_indices, tracking_box_weights)

            tracking_bboxes = tracking_results[:, [2, 3, 4, 5, 1]]

            detection_bboxes = _remap_bboxes(detection_bboxes[0, :, :],
                                             padded_w, padded_h, expand,
                                             self.data_shape, self.ratio)
            tracking_bboxes = _remap_bboxes(tracking_bboxes,
                                            padded_w, padded_h, expand,
                                            self.data_shape, self.ratio)
            mx.nd.waitall()

        # set anchors if needed
        if self._anchor_image_shape != (image.shape[:2]):
            self._anchor_image_shape = image.shape[:2]
            # initialize the anchor tensor for assignment
            self.anchor_tensor = anchors[0, :, :]
            half_w = self.anchor_tensor[:, 2] / 2
            half_h = self.anchor_tensor[:, 3] / 2
            center_x = self.anchor_tensor[:, 0].copy()
            center_y = self.anchor_tensor[:, 1].copy()

            # anchors are in the original format of (center_x, center_y, w, h)
            # translate them to (x0, y0, x1, y1)
            self.anchor_tensor[:, 0] = center_x - half_w
            self.anchor_tensor[:, 1] = center_y - half_h

            self.anchor_tensor[:, 2] = center_x + half_w
            self.anchor_tensor[:, 3] = center_y + half_h

            self.anchor_tensor = _remap_bboxes(self.anchor_tensor, padded_w, padded_h, expand,
                                               self.data_shape, self.ratio)
            self._anchor_num = self.anchor_tensor.shape[0]

        return ids[0], scores[0], detection_bboxes, tracking_bboxes, detection_anchor_indices[0].asnumpy()
