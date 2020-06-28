from __future__ import absolute_import, division, print_function

import os
import sys
from tqdm import tqdm
import cv2
import numpy as np
import time

import json

import mxnet as mx
from mxnet import gluon, autograd
import mxnet.numpy as _mx_np
from mxnet.util import is_np_array

import gluoncv.data.kitti as kitti_dataset
from gluoncv.data.kitti.kitti_utils import dict_batchify_fn
from gluoncv.model_zoo import monodepthv2
from gluoncv.model_zoo.monodepthv2.layers import *
from gluoncv.utils.parallel import *
from gluoncv.utils import LRScheduler, LRSequential

from utils import *


class Trainer:
    def __init__(self, options):
        """
        TODO:
            1. model initialization
            2. dataloader
            3. optimization setting
            4. loss function
            5. metrics
        """
        tic = time.time()
        # configuration setting
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        ################### model initialization ###################
        self.models = {}
        self.parameters_to_train = []

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = monodepthv2.ResnetEncoder(
            self.opt.num_layers,
            pretrained=False,  # (self.opt.weights_init == "pretrained"),
            ctx=self.opt.ctx)
        self.parameters_to_train = self.models["encoder"].collect_params()

        self.models["depth"] = monodepthv2.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        # TODO: initialization method should equal to PyTorch implementation
        self.models["depth"].initialize(ctx=self.opt.ctx)
        self.parameters_to_train.update(self.models["depth"].collect_params())

        # debug : using pretrained model
        encoder_path = os.path.join("./models/mono+stereo_640x192_mx", "encoder.params")
        decoder_path = os.path.join("./models/mono+stereo_640x192_mx", "depth.params")
        self.models["encoder"].load_parameters(encoder_path, ctx=self.opt.ctx)
        self.models["depth"].load_parameters(decoder_path, ctx=self.opt.ctx)
        # end

        self.models["encoder"] = DataParallelModel(self.models["encoder"], ctx_list=self.opt.ctx)
        self.models["depth"] = DataParallelModel(self.models["depth"], ctx_list=self.opt.ctx)

        # TODO: use_pose_net for mono training
        if self.use_pose_net:
            exit()

        # TODO: predictive_mask
        if self.opt.predictive_mask:
            exit()

        ################### dataloader ###################
        datasets_dict = {"kitti": kitti_dataset.KITTIRAWDataset,
                         "kitti_odom": kitti_dataset.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        # TODO: move splits file to a common position
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, num_scales=4, is_train=False, img_ext=img_ext)
        self.train_loader = gluon.data.DataLoader(
            train_dataset, batch_size=self.opt.batch_size, shuffle=False,
            batchify_fn=dict_batchify_fn, num_workers=self.opt.num_workers,
            pin_memory=True, last_batch='discard')

        val_dataset = self.dataset(self.opt.data_path, val_filenames,
                                   self.opt.height, self.opt.width,
                                   self.opt.frame_ids, num_scales=4,
                                   is_train=False, img_ext=img_ext)
        self.val_loader = gluon.data.DataLoader(
            val_dataset, batch_size=self.opt.batch_size, shuffle=True,
            batchify_fn=dict_batchify_fn, num_workers=self.opt.num_workers,
            pin_memory=True, last_batch='discard')

        ################### optimization setting ###################
        self.lr_scheduler = LRSequential([
            LRScheduler('step', base_lr=self.opt.learning_rate,
                        nepochs=self.opt.num_epochs, iters_per_epoch=len(train_dataset),
                        step_epoch=[self.opt.scheduler_step_size])
        ])
        optimizer_params = {'lr_scheduler': self.lr_scheduler,
                            'learning_rate': self.opt.learning_rate}

        # TODO: current use ParamterDict.update(); use multiple Trainer to replace it later
        self.optimizer = gluon.Trainer(self.parameters_to_train, 'adam',
                                       optimizer_params)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", "CPU" if self.opt.ctx[0] is mx.cpu() else "GPU")

        ################### loss function ###################
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            # TODO: Multi-GPU (DataParalleCriterion)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(
                self.opt.batch_size, h, w, ctx=self.opt.ctx[0])
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)

        # TODO: Multi-GPU (DataParalleCriterion)

        ################### metrics ###################
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()
        print("init time: ", time.time() - tic)

    def train(self):
        """
        TODO:
            1. run epochs
            2. save model
        """
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            # TODO: save model
            # if (self.epoch + 1) % self.opt.save_frequency == 0:
            #     self.save_model()

    def run_epoch(self):
        """
        TODO:
            1. prediction
            2. compute loss
            3. evaluation
        """
        print("Training")
        tbar = tqdm(self.train_loader)
        for batch_idx, inputs in enumerate(tbar):
            with autograd.record(True):
                before_op_time = time.time()

                outputs, losses = self.process_batch(inputs)
                mx.nd.waitall()
                autograd.backward(losses['loss'])
                exit()
            self.optimizer.step(self.opt.batch_size)

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.as_in_context(self.opt.ctx[0])

        ################### prediction disp ###################
        if self.opt.pose_model_type == "shared":
            pass
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            encoder_outputs = self.models["encoder"](inputs["color_aug", 0, 0])
            features = [x for x in encoder_outputs[0]]
            for i in range(1, len(encoder_outputs)):
                for j in range(len(encoder_outputs[i])):
                    features[j] = mx.nd.concat(
                        features[j],
                        encoder_outputs[i][j].as_in_context(features[j].context),
                        dim=0
                    )
            decoder_outputs = self.models["depth"](features)
            outputs = decoder_outputs[0]
            for i in range(1, len(decoder_outputs)):
                for key in outputs.keys():
                    outputs[key] = mx.nd.concat(
                        outputs[key],
                        decoder_outputs[i][key].as_in_context(outputs[key].context),
                        dim=0
                    )

        ################### image reconstruction ###################
        if self.opt.predictive_mask:
            # TODO: use predictive_mask
            pass

        if self.use_pose_net:
            # TODO: use_pose_net for mono training
            pass

        self.generate_images_pred(inputs, outputs)

        ################### compute loss ###################
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def generate_images_pred(self, inputs, outputs):
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = mx.nd.contrib.BilinearResize2D(disp, height=self.opt.height, width=self.opt.width)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    # TODO: using pose cnn
                    pass

                cam_points = self.backproject_depth[source_scale](depth,
                                                                  inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points,
                                                           inputs[("K", source_scale)],
                                                           T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = mx.nd.BilinearSampler(
                    data=inputs[("color", frame_id, source_scale)],
                    grid=outputs[("sample", frame_id, scale)],
                    name='sampler')

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = mx.nd.abs(target - pred)
        l1_loss = abs_diff.mean(axis=1, keepdims=True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(axis=1, keepdims=True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
          """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = mx.nd.concat(*reprojection_losses, dim=1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = mx.nd.concat(*identity_reprojection_losses, dim=1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = \
                        identity_reprojection_losses.mean(axis=1, keepdims=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # TODO: use predictive_mask
                pass

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(axis=1, keepdims=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += mx.nd.random.randn(
                    *identity_reprojection_loss.shape).as_in_context(
                    identity_reprojection_loss.context) * 0.00001

                combined = mx.nd.concat(identity_reprojection_loss, reprojection_loss, dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise = mx.nd.min(data=combined, axis=1)
                idxs = mx.nd.argmin(data=combined, axis=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).astype('float')

            loss += to_optimise.mean()

            mean_disp = disp.mean(axis=2, keepdims=True).mean(axis=3, keepdims=True)
            norm_disp = disp / (mean_disp + 1e-7)

            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        str_ctr = []
        for ctx in to_save['ctx']:
            str_ctr.append(str(ctx))
        to_save['ctx'] = str_ctr

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)
