"""Directpose implementation"""
# pylint: disable=line-too-long, redefined-builtin, missing-class-docstring, unused-variable,unnecessary-comprehension
import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from ...utils.comm import compute_locations
from ...nn.shape_spec import ShapeSpec
from ...nn.batch_norm import NaiveSyncBatchNorm
from ...nn.deform_conv import DeformConvWithChangeableStride
from ...nn.group_norm import NaiveGroupNorm
from .directpose_outputs import DirectPoseOutputs

__all__ = ["DirectPose"]

INF = 100000000


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


class Predictor(nn.Module):
    def __init__(self, in_channels, kpt_num, kpt_vis):
        super(Predictor, self).__init__()
        channel_per_kpt = 3 if kpt_vis is True else 2
        self.kpt_predictor = DeformConvWithChangeableStride(in_channels, channel_per_kpt * kpt_num, kernel_size=1, padding=0, bias=True)

    def pad_to_target_size(self, features, target_h, target_w):
        N, C, H, W = features.size()
        pad_h = max(0, target_h - H)
        pad_w = max(0, target_w - W)
        return F.pad(features, [0, pad_w, 0, pad_h])

    def forward(self, kpt_bases, sampler_feature, sampled_feature_stride, fpn_stride):
        # assert kpt_bases.size(1) == 2
        assert fpn_stride % sampled_feature_stride == 0
        stride = int(fpn_stride / sampled_feature_stride)
        padded_sampler_feature = self.pad_to_target_size(sampler_feature,
                                                         stride * (kpt_bases.size(2) - 1) + 1,
                                                         stride * (kpt_bases.size(3) - 1) + 1)
        predictor_offsets = self.kpt_predictor(
            padded_sampler_feature,
            torch.cat((kpt_bases[:, 1:2], kpt_bases[:, 0:1]), dim=1) * stride + stride // 2,
            # kpt_bases[:, [1, 0], :, :].contiguous() * stride + stride // 2,
            stride
        )
        return predictor_offsets


class DirectPose(nn.Module):
    """
    Implement DirectPose.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.CONFIG.MODEL.DIRECTPOSE.IN_FEATURES
        self.fpn_strides = cfg.CONFIG.MODEL.DIRECTPOSE.FPN_STRIDES
        self.yield_proposal = cfg.CONFIG.MODEL.DIRECTPOSE.YIELD_PROPOSAL

        self.directpose_head = DIRECTPOSEHead(cfg, [input_shape[f] for f in self.in_features])
        self.in_channels_to_top_module = self.directpose_head.in_channels_to_top_module

        self.directpose_outputs = DirectPoseOutputs(cfg)

    def forward_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]
        logits_pred, bbox_reg_pred, kpt_reg_pred, ctrness_pred, top_feats, kpts_towers, hms = self.directpose_head(
            features, top_module, self.yield_proposal
        )
        return logits_pred, bbox_reg_pred, kpt_reg_pred, ctrness_pred, top_feats, kpts_towers, hms

    def forward(self, images, features, gt_instances=None, top_module=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        logits_pred, bbox_reg_pred, kpt_reg_pred, kpts_locator_reg_pred, ctrness_pred, top_feats, kpts_towers, hms, hms_offset = \
            self.directpose_head(features, top_module, self.yield_proposal)

        results = {}
        if self.yield_proposal:
            results["features"] = {
                f: b for f, b in zip(self.in_features, kpts_towers)
            }

        if isinstance(images, torch.Tensor):
            image_sizes = [image.size()[-2:] for image in images]
        else:
            image_sizes = images.image_sizes

        if self.training:
            results, losses = self.directpose_outputs.losses(
                logits_pred, bbox_reg_pred, kpt_reg_pred, kpts_locator_reg_pred, ctrness_pred, hms, hms_offset,
                locations, gt_instances, top_feats, None
            )

            if self.yield_proposal:
                with torch.no_grad():
                    results["proposals"] = self.directpose_outputs.predict_proposals(
                        logits_pred, bbox_reg_pred, kpt_reg_pred, ctrness_pred, hms,
                        locations, image_sizes, top_feats
                    )
            return results, losses
        else:
            results = self.directpose_outputs.predict_proposals(
                logits_pred, bbox_reg_pred, kpt_reg_pred, ctrness_pred, hms, hms_offset,
                locations, image_sizes, top_feats, None
            )

            return results, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations


class DIRECTPOSEHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.CONFIG.MODEL.DIRECTPOSE.NUM_CLASSES
        self.num_kpts = cfg.CONFIG.MODEL.DIRECTPOSE.NUM_KPTS
        self.fpn_strides = cfg.CONFIG.MODEL.DIRECTPOSE.FPN_STRIDES
        self.hm_loss_type = cfg.CONFIG.MODEL.DIRECTPOSE.HM_LOSS_TYPE
        self.sample_feature = cfg.CONFIG.MODEL.DIRECTPOSE.SAMPLE_FEATURE
        self.groups = cfg.CONFIG.MODEL.DIRECTPOSE.KPALIGN_GROUPS
        self.seperate_conv_feature = cfg.CONFIG.MODEL.DIRECTPOSE.SEPERATE_CONV_FEATURE
        self.seperate_conv_channel = cfg.CONFIG.MODEL.DIRECTPOSE.SEPERATE_CONV_CHANNEL
        self.enable_bbox_branch = cfg.CONFIG.MODEL.DIRECTPOSE.ENABLE_BBOX_BRANCH
        self.hm_channels = cfg.CONFIG.MODEL.DIRECTPOSE.HM_CHANNELS
        self.loss_on_locator = cfg.CONFIG.MODEL.DIRECTPOSE.LOSS_ON_LOCATOR
        self.predict_hm_offset = cfg.CONFIG.MODEL.DIRECTPOSE.HM_OFFSET
        self.enable_kpt_vis_branch = cfg.CONFIG.MODEL.DIRECTPOSE.KPT_VIS
        self.enable_hm_branch = cfg.CONFIG.MODEL.DIRECTPOSE.ENABLE_HM_BRANCH
        self.center_branch = cfg.CONFIG.MODEL.DIRECTPOSE.CENTER_BRANCH
        self.kpt_per_group = [5, 1, 2, 1, 2, 1, 2, 1, 2]
        self.kpt_index = torch.tensor([0, 1, 2, 3, 4, 5, 8, 6, 9, 7, 10, 11, 14, 12, 15, 13, 16])

        head_configs = {"cls": (cfg.CONFIG.MODEL.DIRECTPOSE.NUM_CLS_CONVS,
                                cfg.CONFIG.MODEL.DIRECTPOSE.USE_DEFORMABLE),
                        "kpt": (cfg.CONFIG.MODEL.DIRECTPOSE.NUM_KPT_CONVS,
                                cfg.CONFIG.MODEL.DIRECTPOSE.USE_DEFORMABLE),
                        "share": (cfg.CONFIG.MODEL.DIRECTPOSE.NUM_SHARE_CONVS,
                                  False)}
        if self.enable_bbox_branch:
            head_configs["bbox"] = (cfg.CONFIG.MODEL.DIRECTPOSE.NUM_BOX_CONVS,
                                    cfg.CONFIG.MODEL.DIRECTPOSE.USE_DEFORMABLE)
        if self.enable_hm_branch:
            head_configs["hm"] = (cfg.CONFIG.MODEL.DIRECTPOSE.NUM_HMS_CONVS,
                                  cfg.CONFIG.MODEL.DIRECTPOSE.USE_DEFORMABLE)

        norm = None if cfg.CONFIG.MODEL.DIRECTPOSE.NORM == "none" else cfg.CONFIG.MODEL.DIRECTPOSE.NORM
        self.num_levels = len(input_shape)

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]
        self.in_channels_to_top_module = in_channels

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            # pylint: disable=no-else-raise
            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:
                    raise NotImplementedError("Deformable Conv v2 is not supported yet!")
                    # conv_func = DFConv2d
                else:
                    conv_func = nn.Conv2d

                in_channels = self.in_channels_to_top_module
                out_channels = in_channels
                groups = 1
                num_groups_gn = 32

                if head == 'hm':
                    in_channels = self.in_channels_to_top_module if i == 0 else self.hm_channels
                    out_channels = self.hm_channels
                elif head == 'kpt' and self.seperate_conv_feature:
                    in_channels = self.in_channels_to_top_module if i == 0 else self.seperate_conv_channel * self.groups
                    out_channels = self.seperate_conv_channel * self.groups
                    groups = 1 if i == 0 else self.groups
                    num_groups_gn = out_channels // 4

                tower.append(conv_func(
                    in_channels, out_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=True, groups=groups
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(num_groups_gn, out_channels))
                elif norm == "NaiveGN":
                    tower.append(NaiveGroupNorm(num_groups_gn, out_channels))
                elif norm == "BN":
                    tower.append(ModuleListDial([
                        nn.BatchNorm2d(out_channels) for _ in range(self.num_levels)
                    ]))
                elif norm == "SyncBN":
                    tower.append(ModuleListDial([
                        NaiveSyncBatchNorm(out_channels) for _ in range(self.num_levels)
                    ]))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        in_channels = self.in_channels_to_top_module

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )

        if self.enable_bbox_branch:
            self.bbox_pred = nn.Conv2d(
                in_channels, 4, kernel_size=3,
                stride=1, padding=1
            )

        if self.seperate_conv_feature:
            kpalign_inchannel = self.seperate_conv_channel * self.groups
            conv_groups = self.groups
            num_groups_gn = kpalign_inchannel // 4
        else:
            kpalign_inchannel = in_channels
            conv_groups = 1
            num_groups_gn = 32

        ctrness_inchannel = kpalign_inchannel if self.center_branch == 'kpt' else in_channels
        self.ctrness = nn.Conv2d(ctrness_inchannel, 1, kernel_size=3, stride=1, padding=1)

        self.locator = nn.Conv2d(kpalign_inchannel, 2 * self.groups, kernel_size=3, stride=1, padding=1, groups=conv_groups)
        self.sampler = nn.Sequential(
            nn.Conv2d(kpalign_inchannel, kpalign_inchannel, kernel_size=3, stride=1, padding=1, groups=conv_groups),
            nn.GroupNorm(num_groups_gn, kpalign_inchannel),
            nn.ReLU())
        self.predictor = nn.ModuleList()
        for g in self.kpt_per_group:
            self.predictor.append(Predictor(self.seperate_conv_channel, g, self.enable_kpt_vis_branch))

        if self.enable_hm_branch:
            self.hm_pred = nn.Conv2d(self.hm_channels, self.num_kpts, kernel_size=3, stride=1, padding=1)
            if self.predict_hm_offset:
                self.hm_offset_pred = nn.Conv2d(self.hm_channels, 2, kernel_size=3, stride=1, padding=1)

        if cfg.CONFIG.MODEL.DIRECTPOSE.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(self.num_levels)])
        else:
            self.scales = None

        all_modules = [
            self.cls_tower, self.share_tower, self.kpt_tower,
            self.cls_logits, self.ctrness,
            self.locator, self.sampler, self.predictor,
        ]
        if self.enable_bbox_branch:
            all_modules.append(self.bbox_tower)
            all_modules.append(self.bbox_pred)
        if self.enable_hm_branch:
            all_modules.append(self.hm_tower)
            all_modules.append(self.hm_pred)
            if self.predict_hm_offset:
                all_modules.append(self.hm_offset_pred)

        for modules in all_modules:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.CONFIG.MODEL.DIRECTPOSE.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        if self.enable_hm_branch and self.hm_loss_type == 'focal':
            torch.nn.init.constant_(self.hm_pred.bias, bias_value)

        # if self.enable_kpt_vis_branch:
        #     prior_prob = 0.5
        #     bias_value = -math.log((1 - prior_prob) / prior_prob)
        #     torch.nn.init.constant_(self.kpt_pred.vis_conv.bias, bias_value)

    def forward(self, x, top_module=None, yield_kpts_towers=False):
        logits = []
        bbox_reg = []
        kpts_reg = []
        kpts_locator_reg = []
        ctrness = []
        top_feats = []
        kpt_towers = []
        all_sampler_features = []
        hms = None
        hms_offset = None

        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            kpt_tower = self.kpt_tower(feature)
            if yield_kpts_towers:
                kpt_towers.append(kpt_tower)

            if self.enable_hm_branch and l == 0:
                hm_tower = self.hm_tower(feature)
                hms = self.hm_pred(hm_tower)
                if self.predict_hm_offset:
                    hms_offset = self.hm_offset_pred(hm_tower)

            logits.append(self.cls_logits(cls_tower))

            if self.enable_bbox_branch:
                bbox_tower = self.bbox_tower(feature)
                cur_bbox_reg = self.bbox_pred(bbox_tower)

            if self.center_branch == 'cls':
                center_tower = cls_tower
            elif self.center_branch == 'bbox':
                center_tower = bbox_tower
            else:
                center_tower = kpt_tower
            ctrness.append(self.ctrness(center_tower))

            cur_kpt_locator_offset = self.locator(kpt_tower)
            sampler_feature = self.sampler(kpt_tower)
            b, _, h, w = sampler_feature.shape
            sampler_feature = sampler_feature.reshape(b, self.groups, self.seperate_conv_channel, h, w)
            all_sampler_features.append(sampler_feature)

            cur_kpt_reg = []
            for group_id in range(self.groups):
                per_group_kpt_bases = cur_kpt_locator_offset[:, 2 * group_id:2 * group_id + 2]
                if l == 0:
                    sampler_features = all_sampler_features[0][:, group_id]
                    sampler_features_stride = self.fpn_strides[0]
                else:
                    sampler_features = all_sampler_features[l - 1][:, group_id]
                    sampler_features_stride = self.fpn_strides[l - 1]

                per_group_kps_offsets = self.predictor[group_id](
                    per_group_kpt_bases, sampler_features,
                    sampler_features_stride, self.fpn_strides[l]
                ).reshape(b, self.kpt_per_group[group_id], -1, h, w)
                per_group_kps_offsets = torch.cat(
                    [per_group_kps_offsets[:, :, 0:2, :, :] / float(self.fpn_strides[l] / sampler_features_stride) + per_group_kpt_bases[:, None],
                     per_group_kps_offsets[:, :, 2:, :, :]], dim=2)
                # per_group_kps_offsets[:, :, 0:2, :, :] /= float(self.fpn_strides[l] / sampler_features_stride)
                # per_group_kps_offsets[:, :, 0:2, :, :] += per_group_kpt_bases[:, None]
                cur_kpt_reg.append(per_group_kps_offsets)

            cur_kpt_reg = torch.cat(cur_kpt_reg, dim=1)
            kpt_index = self.kpt_index.to(cur_kpt_reg.device)[None, :, None, None, None].repeat(b, 1, cur_kpt_reg.shape[2], h, w)
            cur_kpt_reg = cur_kpt_reg.gather(1, kpt_index)

            if self.scales is not None:
                if self.enable_bbox_branch:
                    cur_bbox_reg = self.scales[l](cur_bbox_reg)
                if self.loss_on_locator:
                    cur_kpt_locator_offset = self.scales[l](cur_kpt_locator_offset)
                cur_kpt_reg[:, :, 0:2, :, :] = self.scales[l](cur_kpt_reg[:, :, 0:2, :, :])
            cur_kpt_reg = cur_kpt_reg.reshape(b, -1, h, w)

            # Note that we use relu, as in the improved FCOS, instead of exp.
            if self.enable_bbox_branch:
                bbox_reg.append(F.relu(cur_bbox_reg))
            if self.loss_on_locator:
                kpts_locator_reg.append(cur_kpt_locator_offset)
            kpts_reg.append(cur_kpt_reg)

            if top_module is not None:
                top_feats.append(top_module(kpt_tower))

        return logits, bbox_reg, kpts_reg, kpts_locator_reg, ctrness, top_feats, kpt_towers, hms, hms_offset
