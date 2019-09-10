"""Mask Target Generator."""
from __future__ import absolute_import

from mxnet import gluon, autograd
import numpy as np
import mxnet as mx
import pdb


class MaskTargetGenerator(gluon.HybridBlock):
#class MaskTargetGenerator(gluon.Block):
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

    def __init__(self, num_images, num_rois, num_classes, mask_size, use_mask_ratio=False, **kwargs):
        super(MaskTargetGenerator, self).__init__(**kwargs)
        self._num_images = num_images
        self._num_rois = num_rois
        self._num_classes = num_classes
        self._mask_size = mask_size
        self._use_mask_ratio = use_mask_ratio

    def _get_maskiou_ratio(self, F, roi, cls_target, gt_mask, match):
        #wu     did not use gpu since it would result in out of gpu memory 
        
        """
        roi_cpu = roi.copyto(mx.cpu()).astype('int32').asnumpy()  # (512, 4)
        
        cls_target_cpu = cls_target.copyto(mx.cpu()).asnumpy()
        # gt_mask_cpu: (N, 1, H, W)   N: 有多少 gt object, H，W 原图大小
        gt_mask_cpu = gt_mask.copyto(mx.cpu()).asnumpy()
        gt_mask_area_cpu = gt_mask_cpu.sum((1,2,3))
        match_cpu = match.copyto(mx.cpu()).astype('int32').asnumpy()
        """

        #roi = F.round(roi)

        roi_cpu = roi.asnumpy().astype(np.int32)  # (512, 4)
        
        cls_target_cpu = cls_target.asnumpy()
        # gt_mask_cpu: (N, 1, H, W)   N: 有多少 gt object, H，W 原图大小
        gt_mask_cpu = gt_mask.asnumpy()
        gt_mask_area_cpu = gt_mask_cpu.sum((1,2,3))
        match_cpu = match.asnumpy().astype(np.int32)
        
        """
        roi_cpu = roi.astype(np.int32)  # (512, 4)
        #roi_cpu = F.round(roi)  # (512, 4)
        
        cls_target_cpu = cls_target
        # gt_mask_cpu: (N, 1, H, W)   N: 有多少 gt object, H，W 原图大小
        gt_mask_cpu = gt_mask
        gt_mask_area_cpu = gt_mask_cpu.sum((1,2,3))
        match_cpu = match.astype(np.int32)
        """


        mask_ratios = []
        for ind in range(len(cls_target)):
            #wu gt_mask: (12, 1, 824, 1067)

            #pdb.set_trace()
            #match_id = F.slice(match, begin=(ind,), end=(ind+1,), step=(1,))
            #roi_begin_0 = F.slice(roi, begin=())
            #mask_inside =  F.slice(gt_mask, begin=(,None, , ), end=(,None, , ), step=(1,1,1,1) )

            """
            if cls_target_cpu[ind].asscalar()>0:
                mask_inside = gt_mask_cpu[match_cpu[ind].asscalar(), :, roi_cpu[ind,1].asscalar():roi_cpu[ind, 3].asscalar(), roi_cpu[ind, 0].asscalar():roi_cpu[ind, 2].asscalar()]
                
                mask_inside_sum = mask_inside.sum()
                #pdb.set_trace()
                mask_full_sum = gt_mask_area_cpu[match_cpu[ind]]
                mask_ratio = mask_inside_sum / (mask_full_sum + 1e-7)
                mask_ratios.append( mask_ratio.asscalar() )
            else:
                #pdb.set_trace()
                #mask_ratios.append(0)
                #temp = [F.zeros(1)] * (512 - ind)
                temp = [0] * (512 - ind )
                mask_ratios += temp
                break
            """

            if cls_target_cpu[ind]>0:
                mask_inside = gt_mask_cpu[match_cpu[ind], :, roi_cpu[ind,1]:roi_cpu[ind, 3]+1, roi_cpu[ind, 0]:roi_cpu[ind, 2]+1]
                
                mask_inside_sum = mask_inside.sum()
                mask_full_sum = gt_mask_area_cpu[match_cpu[ind]]
                mask_ratio = mask_inside_sum / (mask_full_sum + 1e-7)
                mask_ratios.append( mask_ratio )
            else:
                #pdb.set_trace()
                mask_ratios.append(0)
                #temp = [0] * (512 - ind)
                #mask_ratios += temp
                #break

        #wu     to_gpu
        #mask_ratios_numpy = np.asarray(mask_ratios)
        mask_ratios = F.array(mask_ratios, ctx=cls_target.context)  # mask_ratios is on gpu? why?  
        #pdb.set_trace()

        return mask_ratios

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, rois, gt_masks, matches, cls_targets, mask_preds):
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

        #F = mx.nd

        # cannot know M (num_gt) to have accurate batch id B * M, must split batch dim
        def _split(x, axis, num_outputs, squeeze_axis):
            x = F.split(x, axis=axis, num_outputs=num_outputs, squeeze_axis=squeeze_axis)
            if isinstance(x, list):
                return x
            elif self._num_images > 1:
                return list(x)
            else:
                return [x]

        with autograd.pause():
            # gt_masks (B, M, H, W) -> (B, M, 1, H, W) -> B * (M, 1, H, W)
            gt_masks = gt_masks.reshape((0, -4, -1, 1, 0, 0))
            gt_masks = _split(gt_masks, axis=0, num_outputs=self._num_images, squeeze_axis=True)
            # rois (B, N, 4) -> B * (N, 4)
            rois = _split(rois, axis=0, num_outputs=self._num_images, squeeze_axis=True)
            # remove possible -1 match
            matches = F.relu(matches)
            # matches (B, N) -> B * (N,)
            matches = _split(matches, axis=0, num_outputs=self._num_images, squeeze_axis=True)
            # cls_targets (B, N) -> B * (N,)
            cls_targets = _split(cls_targets, axis=0, num_outputs=self._num_images,
                                 squeeze_axis=True)

            # mask_pred (B, N, C, MS, MS) -> B * (N, C, MS, MS) 
            mask_preds = _split(mask_preds, axis=0, num_outputs=self._num_images, squeeze_axis=True)

            # (1, C)
            cids = F.arange(1, self._num_classes + 1)
            cids = cids.reshape((1, -1))

            mask_targets = []
            mask_masks = []
            mask_score_targets = []
            mask_score_masks = []
            for roi, gt_mask, match, cls_target, mask_pred in zip(rois, gt_masks, matches, cls_targets, mask_preds):
                # batch id = match
                padded_rois = F.concat(match.reshape((-1, 1)), roi, dim=-1)
                # pooled_mask (N, 1, MS, MS)
                pooled_mask = F.contrib.ROIAlign(gt_mask, padded_rois,
                                                 self._mask_size, 1.0, sample_ratio=2)

                # For mask score 
                #TODO: to add comments
                #TODO: over to intersection
                #TODO: check shape for pooled_mask below when multi-input size = 2
                cls_target_object = F.where(cls_target>0, cls_target-1, F.zeros_like(cls_target))
                selected_index = F.arange(self._num_rois)
                indices = F.stack(selected_index, cls_target_object, axis=0)
                selected_mask = F.gather_nd(mask_pred, indices) 
                #pdb.set_trace()
                selected_mask = selected_mask.reshape((-4, -1, 1, 0, 0))
                #selected_mask = selected_mask.expand_dims(selected_mask, 1)
                # selected_mask: (512, 1, 28, 28)

                pooled_mask_one   =  (pooled_mask > 0) 
                selected_mask_one =  (selected_mask > 0)               

                mask_intersection = pooled_mask_one * selected_mask_one 
                mask_intersection_area = mask_intersection.sum([1,2,3])

                pooled_mask_one_area   = pooled_mask_one.sum([1,2,3])
                selected_mask_one_area =  selected_mask_one.sum([1,2,3])

                if self._use_mask_ratio:
                    mask_ratios = self._get_maskiou_ratio(F, roi, cls_target, gt_mask, match)
                    #pdb.set_trace()
                    pooled_mask_one_area_full =  pooled_mask_one_area / (mask_ratios + 1e-7)
                else:
                    pooled_mask_one_area_full =  pooled_mask_one_area 


                mask_union_area = selected_mask_one_area + pooled_mask_one_area_full - mask_intersection_area

                mask_union_area_final =  F.where((mask_union_area>0), mask_union_area, F.ones_like(mask_union_area))
                maskiou_targets = mask_intersection_area / mask_union_area_final
                #wu maskout_targets: (512, 1)
                maskiou_targets = maskiou_targets.reshape((-4, -1, 1))
                # remove very small value(bias)
                maskiou_targets_flag = (maskiou_targets > 0.05)



                # collect targets
                # (N, 1, MS, MS) -> (N, C, MS, MS)
                mask_target       = F.broadcast_axis(pooled_mask,     size=self._num_classes, axis=1)
                mask_score_target = F.broadcast_axis(maskiou_targets, size=self._num_classes, axis=1)
                #wu maskout_targets: (512, 80)

                # (N,) -> (1, C) -> (N, C, 1, 1)
                cls_target = F.expand_dims(cls_target, 1)
                same_cids = F.broadcast_equal(cls_target, cids)
                same_cids = same_cids.reshape((-2, 1, 1))

                # (N, MS, MS) -> (N, C, 1, 1) -> (N, C, MS, MS)
                mask_mask = F.broadcast_like(same_cids, pooled_mask,
                                             lhs_axes=(2, 3), rhs_axes=(2, 3))
                
                cls_target = cls_target  * maskiou_targets_flag
                mask_score_mask = F.broadcast_equal(cls_target, cids)

                mask_targets.append(mask_target)
                mask_masks.append(mask_mask)
                mask_score_targets.append(mask_score_target)
                mask_score_masks.append(mask_score_mask)

            # B * (N, C, MS, MS) -> (B, N, C, MS, MS)
            mask_targets = F.stack(*mask_targets, axis=0)
            mask_masks = F.stack(*mask_masks, axis=0)
            mask_score_targets = F.stack(*mask_score_targets, axis=0)
            mask_score_masks = F.stack(*mask_score_masks, axis=0) 

        return mask_targets, mask_masks, mask_score_targets, mask_score_masks


