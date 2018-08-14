"""Mask Target Generator."""
from __future__ import absolute_import

from mxnet import gluon


class MaskTargetGenerator(gluon.HybridBlock):
    """Mask RCNN target encoder to generate mask targets.

    Parameters
    ----------
    num_images : int
        Number of input images.
    num_rois : int
        Number of sampled rois.
    num_classes : int
        Number of classes for class-specific targets.
    mask_size : tuple of int
        Size of generated masks, for example (14, 14).

    """
    def __init__(self, num_images, num_rois, num_classes, mask_size, **kwargs):
        super(MaskTargetGenerator, self).__init__(**kwargs)
        self._num_images = num_images
        self._num_rois = num_rois
        self._num_classes = num_classes
        self._mask_size = mask_size

    #pylint: disable=arguments-differ
    def hybrid_forward(self, F, rois, gt_masks, matches, cls_targets):
        """Handle B=self._num_image by a for loop.
        There is no way to know number of gt_masks.

        Parameters
        ----------
        rois: (B, N, 4), input proposals
        gt_masks: (B, M, H, W), input masks of full image size
        matches: (B, N), value [0, M), index to gt_label and gt_box.
        cls_targets: (B, N), value [0, num_class), excluding background class.

        Returns
        -------
        mask_targets: (B, N, C, MS, MS), sampled masks.
        box_target: (B, N, C, 4), only foreground class has nonzero target.
        box_weight: (B, N, C, 4), only foreground class has nonzero weight.

        """
        # cannot know M (num_gt) to have accurate batch id B * M, must split batch dim
        def _split(x, axis, num_outputs, squeeze_axis):
            x = F.split(x, axis=axis, num_outputs=num_outputs, squeeze_axis=squeeze_axis)
            if isinstance(x, list):
                return x
            else:
                return [x]

        # gt_masks (B, M, H, W) -> (B, M, 1, H, W) -> B * (M, 1, H, W)
        gt_masks = gt_masks.reshape((0, -4, -1, 1, 0, 0))
        gt_masks = _split(gt_masks, axis=0, num_outputs=self._num_images, squeeze_axis=True)
        # rois (B, N, 4) -> B * (N, 4)
        rois = _split(rois, axis=0, num_outputs=self._num_images, squeeze_axis=True)
        # remove possible -1 match
        matches = F.where(matches >= 0, matches, F.zeros_like(matches))
        # matches (B, N) -> B * (N,)
        matches = _split(matches, axis=0, num_outputs=self._num_images, squeeze_axis=True)
        # cls_targets (B, N) -> B * (N,)
        cls_targets = _split(cls_targets, axis=0, num_outputs=self._num_images, squeeze_axis=True)

        mask_targets = []
        mask_masks = []
        for roi, gt_mask, match, cls_target in zip(rois, gt_masks, matches, cls_targets):
            # batch id = match
            padded_rois = F.concat(match.reshape((-1, 1)), roi, dim=-1)
            # pooled_mask (N, 1, MS, MS) -> (N, MS, MS)
            pooled_mask = F.contrib.ROIAlign(gt_mask, padded_rois,
                                             self._mask_size, 1.0, sample_ratio=2)
            pooled_mask = pooled_mask.reshape((-3, 0, 0))
            # duplicate to C * (N, MS, MS)
            mask_target = []
            mask_mask = []
            for cid in range(1, self._num_classes + 1):
                # boolean array (N,) -> (N, 1, 1)
                same_cid = (cls_target == cid).reshape((-1, 1, 1))
                # keep orig targets
                mask_target.append(pooled_mask)
                # but mask out the one not belong to this class [N, MS, MS]
                mask_mask.append(F.broadcast_mul(F.ones_like(pooled_mask), same_cid))
            # (C, N, MS, MS) -> (N, C, MS, MS)
            mask_targets.append(F.stack(*mask_target, axis=0).transpose((1, 0, 2, 3)))
            mask_masks.append(F.stack(*mask_mask, axis=0).transpose((1, 0, 2, 3)))

        # B * (N, C, MS, MS) -> (B, N, C, MS, MS)
        mask_targets = F.stack(*mask_targets, axis=0)
        mask_masks = F.stack(*mask_masks, axis=0)
        return mask_targets, mask_masks
