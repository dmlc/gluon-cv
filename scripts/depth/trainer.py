from __future__ import absolute_import, division, print_function

import os
import sys
import shutil
import copy
from tqdm import tqdm

import numpy as np
import json

import mxnet as mx
from mxnet import gluon, autograd

from gluoncv.data import KITTIRAWDataset, KITTIOdomDataset
from gluoncv.data.kitti.kitti_utils import dict_batchify_fn, readlines

from gluoncv.model_zoo import get_model
from gluoncv.model_zoo.monodepthv2.layers import *
from gluoncv.model_zoo.monodepthv2 import MonoDepth2PoseNet
from gluoncv.utils import LRScheduler, LRSequential

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


class Trainer:
    def __init__(self, options, logger):
        # configuration setting
        self.opt = options
        self.logger = logger
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_zoo)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        ######################### dataloader #########################
        datasets_dict = {"kitti": KITTIRAWDataset,
                         "kitti_odom": KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.expanduser("~"), ".mxnet/datasets/kitti",
                             "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, num_scales=4, is_train=True, img_ext=img_ext)
        self.train_loader = gluon.data.DataLoader(
            train_dataset, batch_size=self.opt.batch_size, shuffle=True,
            batchify_fn=dict_batchify_fn, num_workers=self.opt.num_workers,
            pin_memory=True, last_batch='discard')

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, num_scales=4, is_train=False, img_ext=img_ext)
        self.val_loader = gluon.data.DataLoader(
            val_dataset, batch_size=self.opt.batch_size, shuffle=False,
            batchify_fn=dict_batchify_fn, num_workers=self.opt.num_workers,
            pin_memory=True, last_batch='discard')

        ################### model initialization ###################
        # create depth network
        if self.opt.model_zoo is not None:
            self.model = get_model(self.opt.model_zoo, pretrained_base=self.opt.pretrained_base,
                                   scales=self.opt.scales, ctx=self.opt.ctx)
        else:
            assert "Must choose a model from model_zoo, " \
                   "please provide depth the model_zoo using --model_zoo"
        self.logger.info(self.model)

        # resume checkpoint if needed
        if self.opt.resume_depth is not None:
            if os.path.isfile(self.opt.resume_depth):
                logger.info('Resume depth model: %s' % self.opt.resume_depth)
                self.model.load_parameters(self.opt.resume_depth, ctx=self.opt.ctx)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'".format(self.opt.resume_depth))

        if self.use_pose_net:
            # create pose network
            if self.opt.model_zoo_pose is not None:
                self.posenet = get_model(
                    self.opt.model_zoo_pose, pretrained_base=self.opt.pretrained_base,
                    num_input_images=2, num_input_features=1, num_frames_to_predict_for=2,
                    ctx=self.opt.ctx)
            else:
                assert "Must choose a model from model_zoo, " \
                       "please provide the pose model_zoo_pose using --model_zoo_pose"
            self.logger.info(self.posenet)

            # resume checkpoint if needed
            if self.opt.resume_pose is not None:
                if os.path.isfile(self.opt.resume_pose):
                    logger.info('Resume pose model: %s' % self.opt.resume_pose)
                    self.model.load_parameters(self.opt.resume_pose, ctx=self.opt.ctx)
                else:
                    raise RuntimeError("=> no checkpoint found at '{}'".format(
                                        self.opt.resume_pose))

        if self.opt.hybridize:
            self.model.hybridize()
            self.posenet.hybridize()

        ################### optimization setting ###################
        self.lr_scheduler_depth = LRSequential([
            LRScheduler('step', base_lr=self.opt.learning_rate,
                        nepochs=self.opt.num_epochs - self.opt.warmup_epochs,
                        iters_per_epoch=len(self.train_loader),
                        step_epoch=[self.opt.scheduler_step_size - self.opt.warmup_epochs])
        ])
        optimizer_params_depth = {'lr_scheduler': self.lr_scheduler_depth,
                                  'learning_rate': self.opt.learning_rate}

        self.depth_optimizer = gluon.Trainer(self.model.collect_params(), 'adam', optimizer_params_depth)

        if self.use_pose_net:
            self.lr_scheduler_pose = LRSequential([
                LRScheduler('step', base_lr=self.opt.learning_rate,
                            nepochs=self.opt.num_epochs - self.opt.warmup_epochs,
                            iters_per_epoch=len(self.train_loader),
                            step_epoch=[self.opt.scheduler_step_size - self.opt.warmup_epochs])
            ])
            optimizer_params_pose = {'lr_scheduler': self.lr_scheduler_pose,
                                     'learning_rate': self.opt.learning_rate}
            self.pose_optimizer = gluon.Trainer(self.posenet.collect_params(), 'adam', optimizer_params_pose)

        print("Training model named:\n  ", self.opt.model_zoo)
        print("Models are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", "CPU" if self.opt.ctx[0] is mx.cpu() else "GPU")

        ################### loss function ###################
        if not self.opt.no_ssim:
            self.ssim = SSIM()

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(
                self.opt.batch_size, h, w, ctx=self.opt.ctx[0])
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)

        ################### metrics ###################
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

        # for save best model
        self.best_delta1 = 0
        self.best_model = self.model

        if self.use_pose_net:
            self.best_posenet = self.posenet

    def train(self):
        """Run the entire training pipeline
        """
        self.logger.info('Starting Epoch: %d' % self.opt.start_epoch)
        self.logger.info('Total Epochs: %d' % self.opt.num_epochs)

        self.epoch = 0
        for self.epoch in range(self.opt.start_epoch, self.opt.num_epochs):
            self.run_epoch()
            self.val()

        # save final model
        self.save_model("final")
        self.save_model("best")

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        tbar = tqdm(self.train_loader)
        train_loss = 0.0
        for batch_idx, inputs in enumerate(tbar):
            with autograd.record(True):
                outputs, losses = self.process_batch(inputs)
                mx.nd.waitall()

                autograd.backward(losses['loss'])
            self.depth_optimizer.step(self.opt.batch_size, ignore_stale_grad=True)
            if self.use_pose_net:
                self.pose_optimizer.step(self.opt.batch_size, ignore_stale_grad=True)

            train_loss += losses['loss'].asscalar()
            tbar.set_description('Epoch %d, training loss %.3f' %
                                 (self.epoch, train_loss / (batch_idx + 1)))

            if batch_idx % self.opt.log_frequency == 0:
                self.logger.info('Epoch %d iteration %04d/%04d: training loss %.3f' %
                                 (self.epoch, batch_idx, len(self.train_loader),
                                  train_loss / (batch_idx + 1)))
            mx.nd.waitall()

    def process_batch(self, inputs, eval_mode=False):
        for key, ipt in inputs.items():
            inputs[key] = ipt.as_in_context(self.opt.ctx[0])

        # prediction disparity map
        # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        input_img = inputs[("color_aug", 0, 0)]
        if eval_mode:
            input_img = inputs[("color", 0, 0)]

        input_img = input_img.as_in_context(context=self.opt.ctx[0])
        decoder_output = self.model(input_img)

        # for hybridize mode, the output of HybridBlock must is NDArray or List.
        # Here, we have to transfer the output to dict type.
        outputs = {}
        idx = 0
        for i in range(4, -1, -1):
            if i in self.opt.scales:
                outputs[("disp", i)] = decoder_output[idx]
                idx += 1

        if eval_mode:
            _, depth = disp_to_depth(outputs[("disp", 0)],
                                     self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, 0)] = depth

            return outputs

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs))

        # image reconstruction
        self.generate_images_pred(inputs, outputs)

        # compute loss
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def val(self):
        """Validate the model on a single minibatch
        """
        tbar = tqdm(self.val_loader)
        depth_metrics = {}
        abs_rel, sq_rel, rmse, rmse_log = 0, 0, 0, 0
        delta_1, delta_2, delta_3 = 0, 0, 0

        for metric in self.depth_metric_names:
            depth_metrics[metric] = 0
        for i, inputs in enumerate(tbar):
            outputs = self.process_batch(inputs, True)

            if "depth_gt" in inputs:
                self.compute_metrics(inputs, outputs, depth_metrics)

                # print evaluation results
                abs_rel = depth_metrics['de/abs_rel'] / (i + 1)
                sq_rel = depth_metrics['de/sq_rel'] / (i + 1)
                rmse = depth_metrics['de/rms'] / (i + 1)
                rmse_log = depth_metrics['de/log_rms'] / (i + 1)
                delta_1 = depth_metrics['da/a1'] / (i + 1)
                delta_2 = depth_metrics['da/a2'] / (i + 1)
                delta_3 = depth_metrics['da/a3'] / (i + 1)
                tbar.set_description(
                    'Epoch %d, validation '
                    'abs_REL: %.3f sq_REL: %.3f '
                    'RMSE: %.3f, RMSE_log: %.3f '
                    'Delta_1: %.3f Delta_2: %.3f Delta_2: %.3f' %
                    (self.epoch, abs_rel, sq_rel, rmse, rmse_log, delta_1, delta_2, delta_3))
            else:
                print("Cannot find ground truth upon validation dataset!")
                return
        self.logger.info(
            'Epoch %d, validation '
            'abs_REL: %.3f sq_REL: %.3f '
            'RMSE: %.3f, RMSE_log: %.3f '
            'Delta_1: %.3f Delta_2: %.3f Delta_2: %.3f' %
            (self.epoch, abs_rel, sq_rel, rmse, rmse_log, delta_1, delta_2, delta_3))

        mx.nd.waitall()
        if self.epoch % self.opt.save_frequency == 0:
            self.save_checkpoint(delta_1)

        if delta_1 > self.best_delta1:
            self.best_model = self.model
            self.best_delta1 = delta_1
            if self.use_pose_net:
                self.best_posenet = self.posenet

    def predict_poses(self, inputs):
        outputs = {}

        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            if f_i != "s":
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                axisangle, translation = self.posenet(mx.nd.concat(*pose_inputs, dim=1))
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs

    def generate_images_pred(self, inputs, outputs):
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = mx.nd.contrib.BilinearResize2D(disp,
                                                      height=self.opt.height,
                                                      width=self.opt.width)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

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

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(axis=1, keepdims=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss = \
                    identity_reprojection_loss + \
                    mx.nd.random.randn(*identity_reprojection_loss.shape).as_in_context(
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

            loss = loss + self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss = total_loss + loss
            losses["loss/{}".format(scale)] = loss

        total_loss = total_loss / self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_metrics(self, inputs, outputs, depth_metrics):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_gt = inputs["depth_gt"].asnumpy()
        gt_height, gt_width = depth_gt.shape[2:]

        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = mx.nd.clip(
            mx.nd.contrib.BilinearResize2D(depth_pred, height=gt_height, width=gt_width),
            a_min=1e-3, a_max=80
        )
        depth_pred = depth_pred.detach().asnumpy()

        # garg/eigen crop
        mask = depth_gt > 0
        crop_mask = np.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = np.logical_and(mask, crop_mask)

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]

        if self.opt.use_stereo:
            scale_factor = STEREO_SCALE_FACTOR
        else:
            scale_factor = np.median(depth_gt) / np.median(depth_pred)
        depth_pred *= scale_factor

        depth_pred = np.clip(depth_pred, a_min=1e-3, a_max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            depth_metrics[metric] += depth_errors[i]

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

    def save_checkpoint(self, delta_1):
        """Save Checkpoint"""
        save_folder = os.path.join(self.log_path, "models", "weights")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # depth model
        filename = 'epoch_%04d_Delta1_%2.4f.params' % (self.epoch, delta_1)
        filepath = os.path.join(save_folder, filename)
        self.model.save_parameters(filepath)

        # pose encoder model
        if self.use_pose_net:
            filename = 'epoch_%04d_Delta1_%2.4f_posenet.params' % (self.epoch, delta_1)
            filepath = os.path.join(save_folder, filename)
            self.posenet.save_parameters(filepath)

    def save_model(self, model_type="final"):
        """Save Checkpoint"""
        save_folder = os.path.join(self.log_path, "models", "weights")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        model = self.model
        if self.use_pose_net:
            posenet = self.posenet
        if model_type == "best":
            model = self.best_model
            if self.use_pose_net:
                posenet = self.best_posenet

        # save depth model
        filename = 'depth_{}.params'
        filepath = os.path.join(save_folder, filename.format(model_type))
        model.save_parameters(filepath)

        # save pose model
        if self.use_pose_net:
            filename = 'pose_{}.params'
            filepath = os.path.join(save_folder, filename.format(model_type))
            posenet.save_parameters(filepath)
