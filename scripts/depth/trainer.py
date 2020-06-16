from __future__ import absolute_import, division, print_function

import os
import sys
from tqdm import tqdm
import cv2
import numpy as np
import time

import mxnet as mx
from mxnet import gluon
import mxnet.numpy as _mx_np
from mxnet.util import is_np_array

import gluoncv.data.kitti as kitti_dataset
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
            2. optimization setting
            3. dataloader
            4. loss function
            5. metrics
        """

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
            pretrained=(self.opt.weights_init == "pretrained"),
            ctx=self.opt.ctx)
        self.parameters_to_train += list(self.models["encoder"].collect_params())

        self.models["depth"] = monodepthv2.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.parameters_to_train += list(self.models["depth"].collect_params())

        # TODO: use_pose_net for mono training
        if self.use_pose_net:
            exit()

        # TODO: predictive_mask
        if self.opt.predictive_mask:
            exit()

        ################### dataloader ###################
        train_dataset = None

        ################### optimization setting ###################
        self.lr_scheduler = LRSequential([
                LRScheduler('step', base_lr=self.opt.learning_rate,
                            nepochs=self.opt.num_epochs, iters_per_epoch=len(train_dataset),
                            step_epoch=[self.opt.scheduler_step_size])
        ])
        optimizer_params = {'lr_scheduler': self.lr_scheduler,
                            'learning_rate': self.opt.learning_rate}

        self.optimizer = gluon.Trainer(self.parameters_to_train, 'adam',
                                       optimizer_params)

        ################### loss function ###################

        ################### metrics ###################

    def train(self):
        """
        TODO:
            1. run epochs
            2. save model
        """
        pass

    def run_epoch(self):
        """
        TODO:
            1. prediction
            2. compute loss
            3. evaluation
        """
        pass


