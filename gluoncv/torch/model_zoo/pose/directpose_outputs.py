"""Directpose outputs"""
# pylint: disable=line-too-long, redefined-builtin, missing-class-docstring, unused-variable, consider-using-enumerate,unused-argument
import logging
import os
from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import nms

from ...data.structures import Instances, Boxes
from ...utils.comm import get_world_size
from ...nn.focal_loss import sigmoid_focal_loss_jit
from ...nn.smooth_l1_loss import smooth_l1_loss
from ...utils.comm import reduce_sum
from ...nn.nms import ml_nms, oks_nms, close_kpt_nms
from ...nn.iou_loss import IOULoss
from ...nn.keypoint_loss import WeightedMSELoss, HMFocalLoss

logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];

    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets

    ctrness_pred: predicted centerness scores

"""

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
              (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)

class DirectPoseOutputs(nn.Module):
    def __init__(self, cfg):
        super(DirectPoseOutputs, self).__init__()

        self.focal_loss_alpha = cfg.CONFIG.MODEL.DIRECTPOSE.LOSS_ALPHA
        self.focal_loss_gamma = cfg.CONFIG.MODEL.DIRECTPOSE.LOSS_GAMMA
        self.center_sample = cfg.CONFIG.MODEL.DIRECTPOSE.CENTER_SAMPLE
        self.radius = cfg.CONFIG.MODEL.DIRECTPOSE.POS_RADIUS
        self.pre_nms_thresh_train = cfg.CONFIG.MODEL.DIRECTPOSE.INFERENCE_TH_TRAIN
        self.pre_nms_topk_train = cfg.CONFIG.MODEL.DIRECTPOSE.PRE_NMS_TOPK_TRAIN
        self.post_nms_topk_train = cfg.CONFIG.MODEL.DIRECTPOSE.POST_NMS_TOPK_TRAIN
        self.loc_loss_func = IOULoss(cfg.CONFIG.MODEL.DIRECTPOSE.LOC_LOSS_TYPE)
        self.kpt_l1_beta = cfg.CONFIG.MODEL.DIRECTPOSE.KPT_L1_beta
        self.enable_bbox_branch = cfg.CONFIG.MODEL.DIRECTPOSE.ENABLE_BBOX_BRANCH
        self.loss_on_locator = cfg.CONFIG.MODEL.DIRECTPOSE.LOSS_ON_LOCATOR
        self.enable_kpt_vis_branch = cfg.CONFIG.MODEL.DIRECTPOSE.KPT_VIS
        self.enable_close_kpt_nms = cfg.CONFIG.MODEL.DIRECTPOSE.CLOSEKPT_NMS
        # self.vis_res_dir = os.path.join('/workplace/data/motion_efs/home/manchenw/AdelaiDet/visualization',
        #                                 cfg.CONFIG.MODEL.WEIGHTS.split('/')[2])

        self.enable_hm_branch = cfg.CONFIG.MODEL.DIRECTPOSE.ENABLE_HM_BRANCH
        self.hm_type = cfg.CONFIG.MODEL.DIRECTPOSE.HM_TYPE
        self.predict_hm_offset = cfg.CONFIG.MODEL.DIRECTPOSE.HM_OFFSET
        self.hm_loss_type = cfg.CONFIG.MODEL.DIRECTPOSE.HM_LOSS_TYPE
        self.combine_hm_and_kpt = cfg.CONFIG.MODEL.DIRECTPOSE.REFINE_KPT
        self.hm_loss_weight = cfg.CONFIG.MODEL.DIRECTPOSE.HM_LOSS_WEIGHT
        self.g = self.get_gaussian_kernel(sigma=2)
        if self.hm_type == 'Gaussian':
            if self.hm_loss_type == 'mse':
                self.hm_bg_weight = cfg.CONFIG.MODEL.DIRECTPOSE.HM_MSELOSS_BG_WEIGHT
                self.hm_loss_weight = cfg.CONFIG.MODEL.DIRECTPOSE.HM_MSELOSS_WEIGHT
                self.hm_loss = WeightedMSELoss()
            elif self.hm_loss_type == 'focal':
                self.hm_bg_weight = 1.0
                hm_focal_alpha = cfg.CONFIG.MODEL.DIRECTPOSE.HM_FOCALLOSS_ALPHA
                hm_focal_beta = cfg.CONFIG.MODEL.DIRECTPOSE.HM_FOCALLOSS_BETA
                self.hm_loss = HMFocalLoss(hm_focal_alpha, hm_focal_beta)

        self.pre_nms_thresh_test = cfg.CONFIG.MODEL.DIRECTPOSE.INFERENCE_TH_TEST
        self.pre_nms_topk_test = cfg.CONFIG.MODEL.DIRECTPOSE.PRE_NMS_TOPK_TEST
        self.post_nms_topk_test = cfg.CONFIG.MODEL.DIRECTPOSE.POST_NMS_TOPK_TEST
        self.nms_thresh = cfg.CONFIG.MODEL.DIRECTPOSE.NMS_TH
        self.thresh_with_ctr = cfg.CONFIG.MODEL.DIRECTPOSE.THRESH_WITH_CTR

        self.num_classes = cfg.CONFIG.MODEL.DIRECTPOSE.NUM_CLASSES
        self.strides = cfg.CONFIG.MODEL.DIRECTPOSE.FPN_STRIDES
        self.num_kpts = cfg.CONFIG.MODEL.DIRECTPOSE.NUM_KPTS

        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.CONFIG.MODEL.DIRECTPOSE.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi
        self.cnt = 0
        # TVM MODE
        self._tvm_mode = cfg.CONFIG.MODEL.TVM_MODE

    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first

    def _get_ground_truth(self, locations, gt_instances, hm_size=None, images=None):
        num_loc_list = [len(loc) for loc in locations]

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(locations, dim=0)

        training_targets = self.compute_targets_for_locations(
            locations, gt_instances, loc_to_size_range, num_loc_list, images
        )

        training_targets["locations"] = [locations.clone() for _ in range(len(gt_instances))]
        training_targets["im_inds"] = [
            locations.new_ones(locations.size(0), dtype=torch.long) * i for i in range(len(gt_instances))
        ]

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        training_targets["fpn_levels"] = [
            loc.new_ones(len(loc), dtype=torch.long) * level
            for level, loc in enumerate(training_targets["locations"])
        ]

        # we normalize reg_targets by FPN's strides here
        bbox_reg_targets = training_targets["bbox_reg_targets"]
        kpt_reg_targets = training_targets["kpt_reg_targets"]
        for l in range(len(bbox_reg_targets)):
            bbox_reg_targets[l] = bbox_reg_targets[l] / float(self.strides[l])
            kpt_reg_targets[l][:, :, :2] = kpt_reg_targets[l][:, :, :2] / float(self.strides[l])

        if self.enable_hm_branch:
            if self.hm_type == 'Gaussian':
                training_targets["hm_targets"], training_targets["hm_weights"], training_targets["hm_offset"] = \
                    self.generate_gaussian_hm_target(gt_instances, hm_size, self.strides[0], sigma=2)
            elif self.hm_type == 'BinaryLabels':
                training_targets["hm_targets"], training_targets["hm_weights"], training_targets["hm_offset"] = \
                    self.generate_binary_hm_target(locations[0:num_loc_list[0]], gt_instances, hm_size)
            # visualize_hm(training_targets["hm_targets"], training_targets["hm_weights"], images)

        return training_targets

    def get_sample_region(self, boxes, strides, num_loc_list, loc_xs, loc_ys, bitmasks=None, radius=1):
        if bitmasks is not None:
            _, h, w = bitmasks.size()

            ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

            m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
            m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_x = m10 / m00
            center_y = m01 / m00
        else:
            center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
            center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5

        num_gts = boxes.shape[0]
        K = len(loc_xs)
        boxes = boxes[None].expand(K, num_gts, 4)
        center_x = center_x[None].expand(K, num_gts)
        center_y = center_y[None].expand(K, num_gts)
        # center_gt = boxes.new_zeros(boxes.shape)
        center_gts = []
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gts.append(torch.stack([
                torch.where(xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0]),
                torch.where(ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1]),
                torch.where(xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax),
                torch.where(ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax)], dim=2))
            # center_gt[beg:end, :, 0] = torch.where(xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0])
            # center_gt[beg:end, :, 1] = torch.where(ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1])
            # center_gt[beg:end, :, 2] = torch.where(xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax)
            # center_gt[beg:end, :, 3] = torch.where(ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax)
            beg = end
        center_gt = torch.cat(center_gts, dim=0)
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, size_ranges, num_loc_list, images=None):
        labels = []
        bbox_reg_targets = []
        kpt_reg_targets = []
        target_inds = []
        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                bbox_reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                kpt_reg_targets.append(locations.new_zeros((locations.size(0), self.num_kpts, 3)))
                target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)
                continue

            area = targets_per_im.gt_boxes.area()
            kpts = targets_per_im.gt_keypoints.tensor

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            kpt_x_offset = kpts[:, :, 0][None] - xs[:, None, None]
            kpt_y_offset = kpts[:, :, 1][None] - ys[:, None, None]
            kpt_vis = kpts[:, :, 2][None].expand(len(locations), -1, -1)
            kpt_reg_targets_per_im = torch.stack([kpt_x_offset, kpt_y_offset, kpt_vis], dim=3)
            kpt_reg_targets_per_im[kpt_vis == 0] = 0

            if self.center_sample:
                if targets_per_im.has("gt_bitmasks_full"):
                    bitmasks = targets_per_im.gt_bitmasks_full
                else:
                    bitmasks = None
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, num_loc_list, xs, ys,
                    bitmasks=bitmasks, radius=self.radius
                )
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            kpt_reg_targets_per_im = kpt_reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += len(bboxes)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            bbox_reg_targets.append(reg_targets_per_im)
            kpt_reg_targets.append(kpt_reg_targets_per_im)
            target_inds.append(target_inds_per_im)

        return {
            "labels": labels,
            "bbox_reg_targets": bbox_reg_targets,
            "kpt_reg_targets": kpt_reg_targets,
            "target_inds": target_inds
        }

    def losses(self, logits_pred, bbox_reg_pred, kpt_reg_pred, kpts_locator_reg_pred, ctrness_pred, hms, hms_offset,
               locations, gt_instances, top_feats=None, images=None):
        """
        Return the losses from a set of FCOS predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """

        if self.enable_hm_branch:
            training_targets = self._get_ground_truth(locations, gt_instances, hms.shape[2:], images)
        else:
            training_targets = self._get_ground_truth(locations, gt_instances)

        # Collect all logits and regression predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W from slowest to fastest axis.

        instances = Instances((0, 0))
        instances.labels = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.reshape(-1) for x in training_targets["labels"]
        ], dim=0)
        instances.gt_inds = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.reshape(-1) for x in training_targets["target_inds"]
        ], dim=0)
        instances.im_inds = cat([
            x.reshape(-1) for x in training_targets["im_inds"]
        ], dim=0)
        instances.bbox_reg_targets = cat([
            # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
            x.reshape(-1, 4) for x in training_targets["bbox_reg_targets"]
        ], dim=0, )
        instances.kpt_reg_targets = cat([
            # Reshape: (N, Hi, Wi, 17, 3) -> (N*Hi*Wi, 51)
            x.reshape(-1, self.num_kpts * 3) for x in training_targets["kpt_reg_targets"]
        ], dim=0, )
        instances.locations = cat([
            x.reshape(-1, 2) for x in training_targets["locations"]
        ], dim=0)
        instances.fpn_levels = cat([
            x.reshape(-1) for x in training_targets["fpn_levels"]
        ], dim=0)

        instances.logits_pred = cat([
            # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
            x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in logits_pred
        ], dim=0, )

        output_kpt_dim = 3 if self.enable_kpt_vis_branch else 2
        instances.kpt_reg_pred = cat([
            # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
            x.permute(0, 2, 3, 1).reshape(-1, self.num_kpts * output_kpt_dim) for x in kpt_reg_pred
        ], dim=0, )

        instances.ctrness_pred = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.permute(0, 2, 3, 1).reshape(-1) for x in ctrness_pred
        ], dim=0, )

        if len(top_feats) > 0:
            instances.top_feats = cat([
                # Reshape: (N, -1, Hi, Wi) -> (N*Hi*Wi, -1)
                x.permute(0, 2, 3, 1).reshape(-1, x.size(1)) for x in top_feats
            ], dim=0, )
        if self.enable_bbox_branch:
            instances.bbox_reg_pred = cat([
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                x.permute(0, 2, 3, 1).reshape(-1, 4) for x in bbox_reg_pred
            ], dim=0, )
        if self.loss_on_locator:
            instances.kpt_locator_reg_pred = cat([
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                x.permute(0, 2, 3, 1).reshape(-1, self.num_kpts * 2) for x in kpts_locator_reg_pred
            ], dim=0, )

        if self.enable_hm_branch:
            return self.directpose_losses(instances, training_targets["hm_targets"], training_targets["hm_weights"],
                                          training_targets["hm_offset"], hms, hms_offset)
        else:
            return self.directpose_losses(instances)

    def directpose_losses(self, instances, hm_target=None, hm_weight=None, hm_offset_target=None, hm_pred=None, hm_offset_pred=None):
        num_classes = instances.logits_pred.size(1)
        assert num_classes == self.num_classes

        labels = instances.labels.flatten()

        pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # prepare one_hot
        class_target = torch.zeros_like(instances.logits_pred)
        class_target[pos_inds, labels[pos_inds]] = 1

        class_loss = sigmoid_focal_loss_jit(
            instances.logits_pred,
            class_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg

        instances = instances[pos_inds]
        instances.pos_inds = pos_inds

        ctrness_targets = compute_ctrness_targets(instances.bbox_reg_targets)
        ctrness_targets_sum = ctrness_targets.sum()
        loss_denorm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)
        instances.gt_ctrs = ctrness_targets

        if pos_inds.numel() > 0:
            ctrness_loss = F.binary_cross_entropy_with_logits(
                instances.ctrness_pred,
                ctrness_targets,
                reduction="sum"
            ) / num_pos_avg

            kpt_reg_targets_vis = instances.kpt_reg_targets.reshape(-1, self.num_kpts, 3)
            kpt_reg_vis = kpt_reg_targets_vis[:, :, 2] > 0
            pos_kpt = kpt_reg_vis.sum(dim=1)
            kpt_reg_pred = instances.kpt_reg_pred.reshape(len(instances), self.num_kpts, -1)[:, :, 0:2] * kpt_reg_vis[:, :, None]
            kpt_reg_loss = smooth_l1_loss(
                kpt_reg_pred,
                kpt_reg_targets_vis[:, :, 0:2],
                beta=self.kpt_l1_beta,
                reduction="none"
            )
            kpt_reg_loss = kpt_reg_loss.mean(dim=-1).sum(dim=1)
            kpt_reg_loss /= pos_kpt.clamp(min=1.0)
            kpt_reg_loss = (kpt_reg_loss * ctrness_targets * pos_kpt).sum() / (ctrness_targets * pos_kpt).sum().clamp(min=1.0)
            # kpt_reg_loss = kpt_reg_loss.sum() / loss_denorm

            if self.enable_bbox_branch:
                bbox_reg_loss = self.loc_loss_func(
                    instances.bbox_reg_pred,
                    instances.bbox_reg_targets,
                    ctrness_targets
                ) / loss_denorm

            if self.enable_kpt_vis_branch:
                kpt_vis_pred = instances.kpt_reg_pred.reshape(len(instances), self.num_kpts, -1)[:, :, 2]
                kpt_vis_target = kpt_reg_targets_vis[:, :, 2]
                kpt_vis_target[kpt_vis_target > 1] = 1
                total_kpt_pos = reduce_sum(kpt_vis_target.new_tensor([kpt_vis_target.sum()])).item()
                num_pos_kpt_avg = max(total_kpt_pos / num_gpus, 1.0)
                # kpt_vis_loss = sigmoid_focal_loss_jit(
                #     kpt_vis_pred,
                #     kpt_vis_target,
                #     alpha=self.focal_loss_alpha,
                #     gamma=self.focal_loss_gamma,
                #     reduction="sum",
                # ) / num_pos_kpt_avg
                kpt_vis_loss = F.binary_cross_entropy_with_logits(
                    kpt_vis_pred,
                    kpt_vis_target,
                    reduction="sum") / num_pos_kpt_avg

            if self.loss_on_locator:
                kpt_locator_reg_pred = instances.kpt_locator_reg_pred.reshape(-1, self.num_kpts, 2) * kpt_reg_vis[:, :, None]
                kpt_locator_reg_loss = smooth_l1_loss(
                    kpt_locator_reg_pred,
                    kpt_reg_targets_vis[:, :, 0:2],
                    beta=self.kpt_l1_beta,
                    reduction="none"
                )
                kpt_locator_reg_loss = kpt_locator_reg_loss.sum(dim=1).sum(dim=1)
                kpt_locator_reg_loss[pos_kpt > 0] /= pos_kpt[pos_kpt > 0]
                kpt_locator_reg_loss = kpt_locator_reg_loss.sum() / loss_denorm
        else:
            if self.enable_bbox_branch:
                bbox_reg_loss = instances.bbox_reg_pred.sum() * 0
            if self.enable_kpt_vis_branch:
                kpt_vis_loss = instances.kpt_reg_pred.sum() * 0
            if self.loss_on_locator:
                kpt_locator_reg_loss = instances.kpt_reg_pred.sum() * 0
            ctrness_loss = instances.ctrness_pred.sum() * 0
            kpt_reg_loss = instances.kpt_reg_pred.sum() * 0

        if self.enable_hm_branch:
            if self.hm_type == "Gaussian":
                if self.hm_loss_type == "mse":
                    hm_loss = self.hm_loss(hm_pred, hm_target, hm_weight) * self.hm_loss_weight
                elif self.hm_loss_type == "focal":
                    hm_loss = self.hm_loss(hm_pred, hm_target) * self.hm_loss_weight

                if self.predict_hm_offset:
                    hm_offset_target_vis = hm_offset_target.permute(0, 2, 3, 1, 4).reshape(-1, self.num_kpts, 3)
                    hm_offset_vis = hm_offset_target_vis[:, :, 2] > 0
                    hm_offset_pred = hm_offset_pred.permute(0, 2, 3, 1).reshape(-1, 2)[:, None, :].expand(-1, self.num_kpts, -1) \
                                     * hm_offset_vis[:, :, None]
                    pos_kpt = hm_offset_vis.sum(dim=1)
                    hm_offset_loss = smooth_l1_loss(
                        hm_offset_pred,
                        hm_offset_target_vis[:, :, 0:2],
                        beta=self.kpt_l1_beta,
                        reduction="none"
                    )
                    hm_offset_loss = hm_offset_loss.sum(dim=1).sum(dim=1)
                    hm_offset_loss[pos_kpt > 0] /= pos_kpt[pos_kpt > 0]
                    hm_offset_loss = hm_offset_loss.sum() / loss_denorm
            elif self.hm_type == "BinaryLabels":
                pos_joint_inds = torch.nonzero(hm_target.flatten()).squeeze(1)
                num_pos_local = pos_joint_inds.numel()
                total_num_pos = reduce_sum(pos_joint_inds.new_tensor([num_pos_local])).item()
                num_joint_pos_avg = max(total_num_pos / num_gpus, 1.0)
                hm_loss = sigmoid_focal_loss_jit(
                    hm_pred, hm_target, alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma, reduction="sum") / max(num_joint_pos_avg, 1.0) * self.hm_loss_weight

        losses = {
            "loss_directpose_cls": class_loss,
            "loss_directpose_ctr": ctrness_loss,
            "loss_directpose_kpt_loc": kpt_reg_loss,
        }
        if self.enable_bbox_branch:
            losses["loss_directpose_bbox_loc"] = bbox_reg_loss
        if self.loss_on_locator:
            losses["loss_directpose_kpt_locator_loc"] = kpt_locator_reg_loss
        if self.enable_hm_branch:
            losses["loss_directpose_hm"] = hm_loss
            if self.predict_hm_offset:
                losses["loss_directpose_hm_offset"] = hm_offset_loss
        if self.enable_kpt_vis_branch:
            losses["loss_directpose_kpt_vis"] = kpt_vis_loss
        extras = {
            "instances": instances,
            "loss_denorm": loss_denorm,
        }
        return extras, losses

    def predict_proposals(
            self, logits_pred, bbox_reg_pred, kpt_reg_pred, ctrness_pred, hms, hms_offset,
            locations, image_sizes, top_feats=None, images=None
    ):
        if self.training:
            self.pre_nms_thresh = self.pre_nms_thresh_train
            self.pre_nms_topk = self.pre_nms_topk_train
            self.post_nms_topk = self.post_nms_topk_train
        else:
            self.pre_nms_thresh = self.pre_nms_thresh_test
            self.pre_nms_topk = self.pre_nms_topk_test
            self.post_nms_topk = self.post_nms_topk_test

        sampled_boxes = []

        bundle = {
            "l": locations, "o": logits_pred,
            "c": ctrness_pred,
            "kr": kpt_reg_pred, "s": self.strides,
        }
        if self.enable_bbox_branch:
            bundle["r"] = bbox_reg_pred
        if len(top_feats) > 0:
            bundle["t"] = top_feats

        if self._tvm_mode and not self.training:
            forward_for_single_feature_map_func = self.forward_for_single_feature_map_tvm
        else:
            forward_for_single_feature_map_func = self.forward_for_single_feature_map

        for i, per_bundle in enumerate(zip(*bundle.values())):
            # get per-level bundle
            per_bundle = dict(zip(bundle.keys(), per_bundle))
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            l = per_bundle["l"]
            o = per_bundle["o"]
            r = per_bundle["r"] * per_bundle["s"] if "r" in bundle else None
            if self.enable_kpt_vis_branch:
                b, _, h, w = per_bundle["kr"].shape
                kr_reshape = per_bundle["kr"].reshape(b, self.num_kpts, 3, h, w)
                kr = kr_reshape[:, :, 0:2, :, :].reshape(b, -1, h, w) * per_bundle["s"]
                kr_vis = kr_reshape[:, :, 2, :, :]
            else:
                kr = per_bundle["kr"] * per_bundle["s"]
                kr_vis = None
            c = per_bundle["c"]
            t = per_bundle["t"] if "t" in bundle else None

            # DEBUG:
            # sampled_boxes.append((l, o, r, kr, kr_vis))
            # continue

            sampled_boxes.append(
                forward_for_single_feature_map_func(
                    l, o, r, kr, kr_vis, c, image_sizes, t
                )
            )

            for per_im_sampled_boxes in sampled_boxes[-1]:
                per_im_sampled_boxes.fpn_levels = l.new_ones(
                    len(per_im_sampled_boxes), dtype=torch.long
                ) * i

        boxlists = list(zip(*sampled_boxes))
        # ret = [Instances.cat(boxlist) for boxlist in boxlists]
        # output = [x.pred_boxes.tensor for x in ret] + [x.pred_keypoints for x in ret]
        # return output
        # boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        # boxlists = [self.select_over_all_level(Instances.cat(boxlist)) for boxlist in boxlists]
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]

        if self._tvm_mode and not self.training:
            pred_boxes = [boxlist.pred_boxes.tensor for boxlist in boxlists]
            pred_keypoints = [boxlist.pred_keypoints for boxlist in boxlists]
            pred_scores = [boxlist.scores for boxlist in boxlists]
            pred_ids = [boxlist.pred_classes for boxlist in boxlists]
            assert len(pred_ids) == 1, "tvm supports bs==1 only"
            if self.enable_hm_branch and self.combine_hm_and_kpt:
                pred_keypoints = self._refine_kpt(pred_boxes, pred_keypoints, hms, hms_offset, images, topk=40, thresh=0.1)
            # nms_input = torch.cat((, torch.arange(end=int(pred_boxes[0].shape[0])).to(pred_boxes[0].device).unsqueeze(-1)), dim=1)
            # print(nms_input.shape)
            nms_ret = nms(pred_boxes[0], pred_scores[0], self.nms_thresh)
            return pred_ids[0], nms_ret, pred_boxes[0], pred_scores[0], pred_keypoints[0]

        boxlists = self.select_over_all_levels(boxlists)
        if self.enable_hm_branch and self.combine_hm_and_kpt:
            boxlists = self.refine_kpt(boxlists, hms, hms_offset, images, topk=40, thresh=0.1)
        # visualize_kpt_offset(images, boxlists, vis_dir=self.vis_res_dir, cnt=self.cnt)
        # self.cnt += 1
        return boxlists

    def forward_for_single_feature_map_tvm(
            self, locations, logits_pred, bbox_reg_pred, kpt_reg_pred, kpt_vis_pred,
            ctrness_pred, image_sizes, top_feat=None
    ):
        N, C, H, W = logits_pred.shape

        # put in the same format as locations
        logits_pred = logits_pred.view(N, C, H, W).permute(0, 2, 3, 1)
        logits_pred = logits_pred.reshape(N, -1, C).sigmoid()
        kpt_regression = kpt_reg_pred.view(N, self.num_kpts * 2, H, W).permute(0, 2, 3, 1)
        kpt_regression = kpt_regression.reshape(N, -1, self.num_kpts, 2)
        ctrness_pred = ctrness_pred.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness_pred = ctrness_pred.reshape(N, -1).sigmoid()
        if self.enable_bbox_branch:
            box_regression = bbox_reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
            box_regression = box_regression.reshape(N, -1, 4)
        if self.enable_kpt_vis_branch:
            kpt_vis_pred = kpt_vis_pred.permute(0, 2, 3, 1).reshape(N, -1, self.num_kpts).sigmoid()
        if top_feat is not None:
            top_feat = top_feat.view(N, -1, H, W).permute(0, 2, 3, 1)
            top_feat = top_feat.reshape(N, H * W, -1)

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.thresh_with_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]
        # candidate_inds = logits_pred > self.pre_nms_thresh
        # pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        # pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk)

        if not self.thresh_with_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]

        results = []
        for i in range(N):
            assert logits_pred.shape[-1] == 1, "single detection class only"
            per_box_cls = logits_pred[i]
            mask = (per_box_cls <= self.pre_nms_thresh)
            per_box_cls = per_box_cls.masked_fill(mask, 0)
            # per_box_cls = torch.where(mask, per_box_cls, per_box_cls * 0)
            per_class = per_box_cls * 0
            per_class = per_class.masked_fill(mask, -1)
            # per_candidate_inds = candidate_inds[i]
            # per_box_cls = per_box_cls[per_candidate_inds]

            # per_candidate_nonzeros = per_candidate_inds.nonzero()
            # per_box_loc = per_candidate_nonzeros[:, 0]
            # per_class = per_candidate_nonzeros[:, 1]

            per_kpt_regression = kpt_regression[i]
            per_kpt_regression = per_kpt_regression.masked_fill(mask.unsqueeze(-1), 0)
            # per_kpt_regression = per_kpt_regression[per_box_loc]
            # per_locations = locations[per_box_loc]
            per_locations = locations.clone()
            per_locations = per_locations.masked_fill(mask, 0)

            if self.enable_bbox_branch:
                per_box_regression = box_regression[i]
                per_box_regression = per_box_regression.masked_fill(mask, 0)
                # per_box_regression = per_box_regression[per_box_loc]
            if self.enable_kpt_vis_branch:
                per_kpt_vis = kpt_vis_pred[i]
                per_kpt_vis = per_kpt_vis.masked_fill(mask, 0)
                # per_kpt_vis = per_kpt_vis[per_box_loc]
            if top_feat is not None:
                per_top_feat = top_feat[i]
                per_top_feat = per_top_feat.masked_fill(mask, 0)
                # per_top_feat = per_top_feat[per_box_loc]

            # per_pre_nms_top_n = pre_nms_top_n[i]

            # It will only happen when there are more than 1000 person candidate in the image
            # if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
            #     per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
            #     per_class = per_class[top_k_indices]

            #     per_kpt_regression = per_kpt_regression[top_k_indices]
            #     per_locations = per_locations[top_k_indices]
            #     if self.enable_bbox_branch:
            #         per_box_regression = per_box_regression[top_k_indices]
            #     if self.enable_kpt_vis_branch:
            #         per_kpt_vis = per_kpt_vis[top_k_indices]
            #     if top_feat is not None:
            #         per_top_feat = per_top_feat[top_k_indices]

            if self.enable_kpt_vis_branch:
                keypoints = torch.stack([
                    per_locations[:, 0:1] + per_kpt_regression[:, :, 0],
                    per_locations[:, 1:] + per_kpt_regression[:, :, 1],
                    per_kpt_vis], dim=2)
            else:
                keypoints = torch.stack([
                    per_locations[:, 0:1] + per_kpt_regression[:, :, 0],
                    per_locations[:, 1:] + per_kpt_regression[:, :, 1],
                    torch.sqrt(per_box_cls)[:, None].expand(-1, self.num_kpts)], dim=2)

            if self.enable_bbox_branch:
                detections = torch.stack([
                    per_locations[:, 0] - per_box_regression[:, 0],
                    per_locations[:, 1] - per_box_regression[:, 1],
                    per_locations[:, 0] + per_box_regression[:, 2],
                    per_locations[:, 1] + per_box_regression[:, 3],
                ], dim=1)
            else:
                min_xy, _ = keypoints.min(dim=1)
                max_xy, _ = keypoints.max(dim=1)
                detections = torch.cat((min_xy[:, 0:2], max_xy[:, 0:2]), dim=1)
                detections = torch.stack((
                    detections[:, 0].clamp(min=0., max=float(image_sizes[i][1])),
                    detections[:, 1].clamp(min=0., max=float(image_sizes[i][0])),
                    detections[:, 2].clamp(min=0., max=float(image_sizes[i][1])),
                    detections[:, 3].clamp(min=0., max=float(image_sizes[i][0]))
                    ), dim=1)

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.pred_keypoints = keypoints
            boxlist.scores = torch.sqrt(per_box_cls.squeeze(-1))
            boxlist.pred_classes = per_class.squeeze(-1)
            boxlist.locations = per_locations
            if top_feat is not None:
                boxlist.top_feat = per_top_feat
            results.append(boxlist)
            # print(boxlist.pred_boxes.tensor.shape, boxlist.pred_keypoints.shape, boxlist.scores.shape, boxlist.pred_classes.shape, boxlist.locations.shape)

        return results

    def forward_for_single_feature_map(
            self, locations, logits_pred, bbox_reg_pred, kpt_reg_pred, kpt_vis_pred,
            ctrness_pred, image_sizes, top_feat=None
    ):
        N, C, H, W = logits_pred.shape

        # put in the same format as locations
        logits_pred = logits_pred.view(N, C, H, W).permute(0, 2, 3, 1)
        logits_pred = logits_pred.reshape(N, -1, C).sigmoid()
        kpt_regression = kpt_reg_pred.view(N, self.num_kpts * 2, H, W).permute(0, 2, 3, 1)
        kpt_regression = kpt_regression.reshape(N, -1, self.num_kpts, 2)
        ctrness_pred = ctrness_pred.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness_pred = ctrness_pred.reshape(N, -1).sigmoid()
        if self.enable_bbox_branch:
            box_regression = bbox_reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
            box_regression = box_regression.reshape(N, -1, 4)
        if self.enable_kpt_vis_branch:
            kpt_vis_pred = kpt_vis_pred.permute(0, 2, 3, 1).reshape(N, -1, self.num_kpts).sigmoid()
        if top_feat is not None:
            top_feat = top_feat.view(N, -1, H, W).permute(0, 2, 3, 1)
            top_feat = top_feat.reshape(N, H * W, -1)

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.thresh_with_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]
        candidate_inds = logits_pred > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk)

        if not self.thresh_with_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = logits_pred[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero(as_tuple=False)
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_kpt_regression = kpt_regression[i]
            per_kpt_regression = per_kpt_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            if self.enable_bbox_branch:
                per_box_regression = box_regression[i]
                per_box_regression = per_box_regression[per_box_loc]
            if self.enable_kpt_vis_branch:
                per_kpt_vis = kpt_vis_pred[i]
                per_kpt_vis = per_kpt_vis[per_box_loc]
            if top_feat is not None:
                per_top_feat = top_feat[i]
                per_top_feat = per_top_feat[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            # It will only happen when there are more than 1000 person candidate in the image
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]

                per_kpt_regression = per_kpt_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                if self.enable_bbox_branch:
                    per_box_regression = per_box_regression[top_k_indices]
                if self.enable_kpt_vis_branch:
                    per_kpt_vis = per_kpt_vis[top_k_indices]
                if top_feat is not None:
                    per_top_feat = per_top_feat[top_k_indices]

            if self.enable_kpt_vis_branch:
                keypoints = torch.stack([
                    per_locations[:, 0:1] + per_kpt_regression[:, :, 0],
                    per_locations[:, 1:] + per_kpt_regression[:, :, 1],
                    per_kpt_vis], dim=2)
            else:
                keypoints = torch.stack([
                    per_locations[:, 0:1] + per_kpt_regression[:, :, 0],
                    per_locations[:, 1:] + per_kpt_regression[:, :, 1],
                    torch.sqrt(per_box_cls)[:, None].expand(-1, self.num_kpts)], dim=2)

            if self.enable_bbox_branch:
                detections = torch.stack([
                    per_locations[:, 0] - per_box_regression[:, 0],
                    per_locations[:, 1] - per_box_regression[:, 1],
                    per_locations[:, 0] + per_box_regression[:, 2],
                    per_locations[:, 1] + per_box_regression[:, 3],
                ], dim=1)
            else:
                if per_pre_nms_top_n == 0:
                    detections = torch.empty(0, 4).to(keypoints.device)
                else:
                    min_xy, _ = keypoints.min(dim=1)
                    max_xy, _ = keypoints.max(dim=1)
                    detections = torch.cat((min_xy[:, 0:2], max_xy[:, 0:2]), dim=1)
                    detections[:, 0] = detections[:, 0].clamp(min=0, max=image_sizes[i][1])
                    detections[:, 1] = detections[:, 1].clamp(min=0, max=image_sizes[i][0])
                    detections[:, 2] = detections[:, 2].clamp(min=0, max=image_sizes[i][1])
                    detections[:, 3] = detections[:, 3].clamp(min=0, max=image_sizes[i][0])

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.pred_keypoints = keypoints
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            if top_feat is not None:
                boxlist.top_feat = per_top_feat
            results.append(boxlist)

        return results

    def _refine_kpt(self, pred_boxes, pred_keypoints, heatmaps, hms_offset=None, images=None, stride=8, topk=40, thresh=0.2):
        heatmaps = torch.clamp(heatmaps.sigmoid_(), min=1e-4, max=1-1e-4)
        # visualize_hm(heatmaps, None, images)
        heatmaps = _nms(heatmaps)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(heatmaps, K=topk)
        if hms_offset is not None:
            hp_offset = _transpose_and_gather_feat(hms_offset, hm_inds.view(len(pred_keypoints), -1))
            hp_offset = hp_offset.view(len(pred_keypoints), self.num_kpts, topk, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs += 0.5
            hm_ys += 0.5
        hm_xs *= stride
        hm_ys *= stride
        # visualize_hm_xy(hm_xs, hm_ys, hm_score, images, thresh=0.1)

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1)

        for i in range(len(pred_keypoints)):
            bbox_pred = pred_boxes[i]
            kpt_regression = pred_keypoints[i][:, :, 0:2].permute(1, 0, 2).contiguous()
            kpt_regression_score = pred_keypoints[i][:, :, 2].permute(1, 0).contiguous()
            dist = (((kpt_regression[:, :, None, :] - hm_kps[i][:, None, :, :]) ** 2).sum(dim=3) ** 0.5)
            min_dist, min_ind = dist.min(dim=2)
            cur_hm_score = hm_score[i].gather(1, min_ind)
            cur_hm_kps = hm_kps[i].gather(1, min_ind[:, :, None].expand(-1, -1, 2))
            mask = (cur_hm_kps[:, :, 0] < bbox_pred[None, :, 0]) + (cur_hm_kps[:, :, 0] > bbox_pred[None, :, 2]) + \
                   (cur_hm_kps[:, :, 1] < bbox_pred[None, :, 1]) + (cur_hm_kps[:, :, 1] > bbox_pred[None, :, 3]) + \
                   (cur_hm_score < thresh) + \
                   (min_dist > (torch.max(bbox_pred[:, 2] - bbox_pred[:, 0], bbox_pred[:, 3] - bbox_pred[:, 1]) * 0.1)[None])
            mask = (mask > 0).float()
            kps_score = (1 - mask) * cur_hm_score + mask * kpt_regression_score
            mask = mask[:, :, None].expand(-1, -1, 2)
            kps = (1 - mask) * cur_hm_kps + mask * kpt_regression
            kps = torch.cat((kps, kps_score[:, :, None]), dim=2)
            pred_keypoints[i] = kps.permute(1, 0, 2).contiguous()
        return pred_keypoints

    def refine_kpt(self, boxlists, heatmaps, hms_offset=None, images=None, stride=8, topk=40, thresh=0.2):
        heatmaps = torch.clamp(heatmaps.sigmoid_(), min=1e-4, max=1-1e-4)
        # visualize_hm(heatmaps, None, images)
        heatmaps = _nms(heatmaps)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(heatmaps, K=topk)
        if hms_offset is not None:
            hp_offset = _transpose_and_gather_feat(hms_offset, hm_inds.view(len(boxlists), -1))
            hp_offset = hp_offset.view(len(boxlists), self.num_kpts, topk, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs += 0.5
            hm_ys += 0.5
        hm_xs *= stride
        hm_ys *= stride
        # visualize_hm_xy(hm_xs, hm_ys, hm_score, images, thresh=0.1)

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1)

        for i in range(len(boxlists)):
            if len(boxlists[i]) == 0:
                continue
            bbox_pred = boxlists[i].pred_boxes.tensor
            kpt_regression = boxlists[i].pred_keypoints[:, :, 0:2].permute(1, 0, 2).contiguous()
            kpt_regression_score = boxlists[i].pred_keypoints[:, :, 2].permute(1, 0).contiguous()
            dist = (((kpt_regression[:, :, None, :] - hm_kps[i][:, None, :, :]) ** 2).sum(dim=3) ** 0.5)
            min_dist, min_ind = dist.min(dim=2)
            cur_hm_score = hm_score[i].gather(1, min_ind)
            cur_hm_kps = hm_kps[i].gather(1, min_ind[:, :, None].expand(-1, -1, 2))
            mask = (cur_hm_kps[:, :, 0] < bbox_pred[None, :, 0]) + (cur_hm_kps[:, :, 0] > bbox_pred[None, :, 2]) + \
                   (cur_hm_kps[:, :, 1] < bbox_pred[None, :, 1]) + (cur_hm_kps[:, :, 1] > bbox_pred[None, :, 3]) + \
                   (cur_hm_score < thresh) + \
                   (min_dist > (torch.max(bbox_pred[:, 2] - bbox_pred[:, 0], bbox_pred[:, 3] - bbox_pred[:, 1]) * 0.1)[None])
            mask = (mask > 0).float()
            kps_score = (1 - mask) * cur_hm_score + mask * kpt_regression_score
            mask = mask[:, :, None].expand(-1, -1, 2)
            kps = (1 - mask) * cur_hm_kps + mask * kpt_regression
            kps = torch.cat((kps, kps_score[:, :, None]), dim=2)
            boxlists[i].pred_keypoints = kps.permute(1, 0, 2).contiguous()
        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            # # Remove skeleton based NMS because seen from the result it is not helping
            # if self.enable_kpt_vis_branch:
            #     result = oks_nms(result, thresh=0.8, in_vis_thre=0.2)
            # if self.enable_close_kpt_nms:
            #     result = close_kpt_nms(result)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.post_nms_topk > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.post_nms_topk + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

    def select_over_all_level(self, boxlist):
        # multiclass nms
        result = ml_nms(boxlist, self.nms_thresh, fixed_size=False)
        # # Remove skeleton based NMS because seen from the result it is not helping
        # if self.enable_kpt_vis_branch:
        #     result = oks_nms(result, thresh=0.8, in_vis_thre=0.2)
        # if self.enable_close_kpt_nms:
        #     result = close_kpt_nms(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.post_nms_topk > 0:
            cls_scores = result.scores
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(),
                number_of_detections - self.post_nms_topk + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


    def get_gaussian_kernel(self, sigma):
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        return g

    def generate_gaussian_hm_target(self, gt_instances, hm_size, feat_stride=8, sigma=2):
        all_hms = []
        all_hms_weight = []
        all_hms_offset = []
        hm_h, hm_w = hm_size
        g = torch.from_numpy(self.g).float().to(gt_instances[0].gt_keypoints.device)

        for gt in gt_instances:
            hms = torch.zeros((self.num_kpts, hm_h, hm_w), dtype=torch.float32, device=gt.gt_keypoints.device)
            if self.hm_loss_type == 'mse':
                hms_weight = 2 * torch.ones((self.num_kpts, hm_h, hm_w), dtype=torch.float32, device=gt.gt_keypoints.device)
            if self.predict_hm_offset:
                hms_offset = torch.zeros((self.num_kpts, hm_h, hm_w, 3), dtype=torch.float32, device=gt.gt_keypoints.device)

            joints = gt.gt_keypoints.tensor
            for p in joints:
                for idx, pt in enumerate(p):
                    if pt[2] > 0:
                        x = int(pt[0] / feat_stride + 0.5)
                        y = int(pt[1] / feat_stride + 0.5)
                        if x < 0 or y < 0 or x >= hm_w or y >= hm_h:
                            continue

                        ul = int(np.floor(x - 3 * sigma - 1)), int(np.floor(y - 3 * sigma - 1))
                        br = int(np.floor(x + 3 * sigma + 2)), int(np.floor(y + 3 * sigma + 2))

                        c, d = max(0, -ul[0]), min(br[0], hm_w) - ul[0]
                        a, b = max(0, -ul[1]), min(br[1], hm_h) - ul[1]

                        cc, dd = max(0, ul[0]), min(br[0], hm_w)
                        aa, bb = max(0, ul[1]), min(br[1], hm_h)

                        hms[idx, aa:bb, cc:dd] = torch.max(hms[idx, aa:bb, cc:dd], g[a:b, c:d])
                        if self.hm_loss_type == 'mse':
                            hms_weight[idx, aa:bb, cc:dd] = 1.0
                        if self.predict_hm_offset:
                            hms_offset[idx, y, x, 0] = pt[0] / feat_stride - x
                            hms_offset[idx, y, x, 1] = pt[1] / feat_stride - y
                            hms_offset[idx, y, x, 2] = 1.0

            all_hms.append(hms)
            if self.hm_loss_type == 'mse':
                hms_weight[hms_weight == 2] = self.hm_bg_weight
                all_hms_weight.append(hms_weight)
            if self.predict_hm_offset:
                all_hms_offset.append(hms_offset)

        all_hms = torch.stack(all_hms, dim=0)
        if self.hm_loss_type == 'mse':
            all_hms_weight = torch.stack(all_hms_weight, dim=0)
        if self.predict_hm_offset:
            all_hms_offset = torch.stack(all_hms_offset, dim=0)

        return all_hms, all_hms_weight, all_hms_offset

    def generate_binary_hm_target(self, locations, gt_instances, hm_size):
        all_hms = []
        locations_x = locations[:, 0:1]
        locations_y = locations[:, 1:2]
        for gt in gt_instances:
            target = torch.zeros((self.num_kpts, hm_size[0] * hm_size[1]), dtype=torch.float32, device=gt.gt_keypoints.device)

            kps = gt.gt_keypoints.tensor.reshape(-1, 3)
            kps_types = torch.arange(kps.size(0), dtype=torch.int, device=kps.device) % self.num_kpts
            if (kps[:, 2] > 0).sum() > 0:
                visible_kps = kps[kps[:, 2] > 0]
                visible_kps_types = kps_types[kps[:, 2] > 0]

                distances_x = torch.abs(locations_x - visible_kps[:, 0][None])
                distances_y = torch.abs(locations_y - visible_kps[:, 1][None])
                distances = torch.max(distances_x, distances_y)
                _, min_inds = distances.min(dim=0)

                for (kpt_idx, heatmap_loc) in zip(visible_kps_types, min_inds):
                    target[kpt_idx, heatmap_loc] = 1
            all_hms.append(target.reshape(self.num_kpts, hm_size[0], hm_size[1]))
        all_hms = torch.stack(all_hms, dim=0)
        return all_hms, None, None

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk_channel(scores, K=40):
    batch, cate, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cate, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs

def _gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat
