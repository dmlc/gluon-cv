"""02. Monodepth2 training on KITTI dataset
==================================================

This is a tutorial of training MonoDepth2 on KITTI dataset using Gluon CV toolkit.
The readers should have basic knowledge of deep learning and should be familiar with Gluon API.
New users may first go through `A 60-minute Gluon Crash Course <http://gluon-crash-course.mxnet.io/>`_.
You can `Start Training Now`_ or `Dive into Deep`_.

Start Training Now
~~~~~~~~~~~~~~~~~~

.. hint::

    Feel free to skip the tutorial because the training script is self-complete and ready to launch.

    :download:`Download Full Python Script: train.py<../../../scripts/depth/train.py>`

    Example training command::

        python train.py --model_zoo monodepth2_resnet18_kitti_stereo_640x192 --pretrained_base --frame_ids 0 --use_stereo --split eigen_full --log_dir ./tmp/stereo/ --png --gpu 0

    For more training command options, please run ``python train.py -h``
    Please checkout the `model_zoo <../model_zoo/depth.html>`_ for training commands of reproducing the pretrained model.

Dive into Deep
~~~~~~~~~~~~~~
"""
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
import gluoncv

##############################################################################
# Digging into Self-Supervised Monocular Depth Prediction
# -----------------------------
#
# .. image:: https://raw.githubusercontent.com/KuangHaofei/GluonCV_Test/master/monodepthv2/tutorials/monodepth2.png
#     :width: 100%
#     :align: center
#
# (figure credit to `Godard et al. <https://arxiv.org/pdf/1806.01260.pdf>`_ )
#
# Self-Supervised Monocular Depth Estimation (Monodepth2) [Godard19]_ build a
# simple depth model and train it with a self-supervised manner by exploit the
# spatial geometry constrain. The key idea of Monodepth2 is that it build a novel
# reprojection loss, include (1) a minimum reprojection loss, designed to robustly
# handle occlusions, (2) a full-resolution multi-scale sampling method that reduces
# visual artifacts, and (3) an auto-masking loss to ignore training pixels that violate
# camera motion assumptions.
#


##############################################################################
# Monodepth2 Model
# ------------
#
# A simple U-Net architecture is used in Monodepth2, which combines multiple scale
# features with different receptive field sizes. It pools the featuremaps
# into different sizes and then concatenating together after upsampling.
#
# The Encoder module is a ResNet, it is defined as::
#
#     class ResnetEncoder(nn.HybridBlock):
#         def __init__(self, backbone, pretrained, num_input_images=1, ctx=cpu(), **kwargs):
#             super(ResnetEncoder, self).__init__()
#
#             self.num_ch_enc = np.array([64, 64, 128, 256, 512])
#
#             resnets = {'resnet18': resnet18_v1b,
#                        'resnet34': resnet34_v1b,
#                        'resnet50': resnet50_v1s,
#                        'resnet101': resnet101_v1s,
#                        'resnet152': resnet152_v1s}
#
#             if backbone not in resnets:
#                 raise ValueError("{} is not a valid resnet".format(backbone))
#
#             if num_input_images > 1:
#                 pass
#             else:
#                 self.encoder = resnets[backbone](pretrained=pretrained, ctx=ctx, **kwargs)
#
#             if backbone not in ('resnet18', 'resnet34'):
#                 self.num_ch_enc[1:] *= 4
#
#         def hybrid_forward(self, F, input_image):
#             # pylint: disable=unused-argument, missing-function-docstring
#             self.features = []
#             x = (input_image - 0.45) / 0.225
#             x = self.encoder.conv1(x)
#             x = self.encoder.bn1(x)
#             self.features.append(self.encoder.relu(x))
#             self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
#             self.features.append(self.encoder.layer2(self.features[-1]))
#             self.features.append(self.encoder.layer3(self.features[-1]))
#             self.features.append(self.encoder.layer4(self.features[-1]))
#
#             return self.features
#
#
# The Decoder module is a fully convolutional network with skip architecture, it exploit the featuremaps
# in different scale and concatenating together after upsampling. It is defined as::
#
#     class DepthDecoder(nn.HybridBlock):
#         def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1,
#                      use_skips=True):
#             super(DepthDecoder, self).__init__()
#
#             self.num_output_channels = num_output_channels
#             self.use_skips = use_skips
#             self.upsample_mode = 'nearest'
#             self.scales = scales
#
#             self.num_ch_enc = num_ch_enc
#             self.num_ch_dec = np.array([16, 32, 64, 128, 256])
#
#             # decoder
#             with self.name_scope():
#                 self.convs = OrderedDict()
#                 for i in range(4, -1, -1):
#                     # upconv_0
#                     num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
#                     num_ch_out = self.num_ch_dec[i]
#                     self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
#
#                     # upconv_1
#                     num_ch_in = self.num_ch_dec[i]
#                     if self.use_skips and i > 0:
#                         num_ch_in += self.num_ch_enc[i - 1]
#                     num_ch_out = self.num_ch_dec[i]
#                     self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
#
#                 for s in self.scales:
#                     self.convs[("dispconv", s)] = Conv3x3(
#                         self.num_ch_dec[s], self.num_output_channels)
#
#                 # register blocks
#                 for k in self.convs:
#                     self.register_child(self.convs[k])
#                 self.decoder = nn.HybridSequential()
#                 self.decoder.add(*list(self.convs.values()))
#
#                 self.sigmoid = nn.Activation('sigmoid')
#
#         def hybrid_forward(self, F, input_features):
#             # pylint: disable=unused-argument, missing-function-docstring
#             self.outputs = []
#
#             # decoder
#             x = input_features[-1]
#             for i in range(4, -1, -1):
#                 x = self.convs[("upconv", i, 0)](x)
#                 x = [F.UpSampling(x, scale=2, sample_type='nearest')]
#                 if self.use_skips and i > 0:
#                     x += [input_features[i - 1]]
#                 x = F.concat(*x, dim=1)
#                 x = self.convs[("upconv", i, 1)](x)
#                 if i in self.scales:
#                     self.outputs.append(self.sigmoid(self.convs[("dispconv", i)](x)))
#
#             return self.outputs
#
# Monodepth model is provided in :class:`gluoncv.model_zoo.MonoDepth2`. To get
# Monodepth2 model using ResNet18 base network:
model = gluoncv.model_zoo.get_monodepth2(backbone='resnet18')
print(model)


##############################################################################
# Dataset and Data Augmentation
# -----------------------------
#
# - Prepare KITTI RAW Dataset:
#
#     Here we give an example of training monodepth2 on the KITTI RAW dataset [Godard19]_. First,
#     we need to prepare the dataset. The official implementation of monodepth2 does not use all
#     the data of KITTI, here we use the same dataset and split method as it. You need download
#     the split zip file, and extract it to ``$(HOME)/.mxnet/datasets/kitti/``.
#
#     Follow the command to get the dataset::
#
#       cd ~
#       mkdir -p .mxnet/datasets/kitti
#       cd .mxnet/datasets/kitti
#       wget https://github.com/KuangHaofei/GluonCV_Test/raw/master/monodepthv2/tutorials/splits.zip
#       unzip splits.zip
#       wget -i splits/kitti_archives_to_download.txt -P kitti_data/
#       cd kitti_data
#       unzip "*.zip"
#
#  .. hint::
#
#     You need 175GB, free disk space to download and extract this dataset. SSD harddrives are recommended
#     for faster speed. The time it takes to prepare the dataset depends on your Internet connection and
#     disk speed. For example, it takes around 2 hours on an AWS EC2 instance with EBS.
#
# We provide self-supervised depth estimation datasets in :class:`gluoncv.data`.
# For example, we can easily get the KITTI RAW Stereo dataset:
import os
from gluoncv.data.kitti import readlines, dict_batchify_fn

train_filenames = os.path.join(
    os.path.expanduser("~"), '.mxnet/datasets/kitti/splits/eigen_full/train_files.txt')
train_filenames = readlines(train_filenames)
train_dataset = gluoncv.data.KITTIRAWDataset(
    filenames=train_filenames, height=192, width=640,
    frame_idxs=[0, "s"], num_scales=4, is_train=True, img_ext='.png')
print('Training images:', len(train_dataset))
# set batch_size = 12 for toy example
batch_size = 12
train_loader = gluon.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, batchify_fn=dict_batchify_fn,
    num_workers=12, pin_memory=True, last_batch='discard')

##############################################################################
# For data augmentation,
# we follow the standard data augmentation routine to transform the input image.
# Here, we just use RandomFlip with 50% probability for input images.
#
# Random pick one example for visualization:
import random
from datetime import datetime
random.seed(datetime.now())
idx = random.randint(0, len(train_dataset))

data = train_dataset[idx]
input_img = data[("color", 0, 0)]
input_stereo_img = data[("color", 's', 0)]
input_gt = data['depth_gt']

input_img = np.transpose((input_img.asnumpy() * 255).astype(np.uint8), (1, 2, 0))
input_stereo_img = np.transpose((input_stereo_img.asnumpy() * 255).astype(np.uint8), (1, 2, 0))
input_gt = np.transpose((input_gt.asnumpy()).astype(np.uint8), (1, 2, 0))

from PIL import Image
input_img = Image.fromarray(input_img)
input_stereo_img = Image.fromarray(input_stereo_img)
input_gt = Image.fromarray(input_gt[:, :, 0])

input_img.save("input_img.png")
input_stereo_img.save("input_stereo_img.png")
input_gt.save("input_gt.png")

##############################################################################
# Plot the stereo image pairs and ground truth of the left image
from matplotlib import pyplot as plt

input_img = Image.open('input_img.png').convert('RGB')
input_stereo_img = Image.open('input_stereo_img.png').convert('RGB')
input_gt = Image.open('input_gt.png')

fig = plt.figure()
# subplot 1 for left image
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.75)
fig.add_subplot(3, 1, 1)
plt.title("left image")
plt.imshow(input_img)
# subplot 2 for right images
fig.add_subplot(3, 1, 2)
plt.title("right image")
plt.imshow(input_stereo_img)
# subplot 3 for the ground truth
fig.add_subplot(3, 1, 3)
plt.title("ground truth of left input (the reprojection of LiDAR data)")
plt.imshow(input_gt)
# display
plt.show()

##############################################################################
# Training Details
# ----------------
#
# - Training Losses:
#
#     We apply a standard reprojection loss to train Monodepth2.
#     As describes in Monodepth2 [Godard19]_ , the reprojection loss include three parts:
#     a multi-scale reprojection loss (combined L1 loss and SSIM loss), an auto-masking loss and
#     an edge-aware smoothness loss as in Monodepth [Godard17]_ .
#
# The computation of loss is defined as
# (Please checkout the full :download:`trainer.py<../../../scripts/depth/trainer.py>` for complete implementation.)::
#
#     def compute_losses(self, inputs, outputs):
#         """Compute the reprojection and smoothness losses for a minibatch
#         """
#         losses = {}
#         total_loss = 0
#
#         for scale in self.opt.scales:
#             loss = 0
#             reprojection_losses = []
#
#             if self.opt.v1_multiscale:
#                 source_scale = scale
#             else:
#                 source_scale = 0
#
#             disp = outputs[("disp", scale)]
#             color = inputs[("color", 0, scale)]
#             target = inputs[("color", 0, source_scale)]
#
#             for frame_id in self.opt.frame_ids[1:]:
#                 pred = outputs[("color", frame_id, scale)]
#                 reprojection_losses.append(self.compute_reprojection_loss(pred, target))
#
#             reprojection_losses = mx.nd.concat(*reprojection_losses, dim=1)
#
#             if not self.opt.disable_automasking:
#                 identity_reprojection_losses = []
#                 for frame_id in self.opt.frame_ids[1:]:
#                     pred = inputs[("color", frame_id, source_scale)]
#                     identity_reprojection_losses.append(
#                         self.compute_reprojection_loss(pred, target))
#
#                 identity_reprojection_losses = mx.nd.concat(*identity_reprojection_losses, dim=1)
#
#                 if self.opt.avg_reprojection:
#                     identity_reprojection_loss = \
#                         identity_reprojection_losses.mean(axis=1, keepdims=True)
#                 else:
#                     # save both images, and do min all at once below
#                     identity_reprojection_loss = identity_reprojection_losses
#
#             if self.opt.avg_reprojection:
#                 reprojection_loss = reprojection_losses.mean(axis=1, keepdims=True)
#             else:
#                 reprojection_loss = reprojection_losses
#
#             if not self.opt.disable_automasking:
#                 # add random numbers to break ties
#                 identity_reprojection_loss = \
#                     identity_reprojection_loss + \
#                     mx.nd.random.randn(*identity_reprojection_loss.shape).as_in_context(
#                         identity_reprojection_loss.context) * 0.00001
#
#                 combined = mx.nd.concat(identity_reprojection_loss, reprojection_loss, dim=1)
#             else:
#                 combined = reprojection_loss
#
#             if combined.shape[1] == 1:
#                 to_optimise = combined
#             else:
#                 to_optimise = mx.nd.min(data=combined, axis=1)
#                 idxs = mx.nd.argmin(data=combined, axis=1)
#
#             if not self.opt.disable_automasking:
#                 outputs["identity_selection/{}".format(scale)] = (
#                         idxs > identity_reprojection_loss.shape[1] - 1).astype('float')
#
#             loss += to_optimise.mean()
#
#             mean_disp = disp.mean(axis=2, keepdims=True).mean(axis=3, keepdims=True)
#             norm_disp = disp / (mean_disp + 1e-7)
#
#             smooth_loss = get_smooth_loss(norm_disp, color)
#
#             loss = loss + self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
#             total_loss = total_loss + loss
#             losses["loss/{}".format(scale)] = loss
#
#         total_loss = total_loss / self.num_scales
#         losses["loss"] = total_loss
#         return losses

##############################################################################
# - Learning Rate and Scheduling:
#
#     Here, we follow the standard strategy of monodepth2. The network is trained for 20 epochs using Adam.
#     We use a 'step' learning rate scheduler for Monodepth2 training, provided in :class:`gluoncv.utils.LRScheduler`.
#     We use a learning rate of 10−4 for the first 15 epochs which is then dropped to 10−5 for the remainder.
#
lr_scheduler = gluoncv.utils.LRSequential([
    gluoncv.utils.LRScheduler(
        'step', base_lr=1e-4, nepochs=20, iters_per_epoch=len(train_dataset), step_epoch=[15])
])
optimizer_params = {'lr_scheduler': lr_scheduler,
                    'learning_rate': 1e-4}

##############################################################################
# - Create Adam solver
optimizer = gluon.Trainer(model.collect_params(), 'adam', optimizer_params)

##############################################################################
# The training loop
# -----------------
#
# Please checkout the full :download:`trainer.py<../../../scripts/depth/trainer.py>` for complete implementation.
# This is an example of training loop::
#
#     def train(self):
#         """Run the entire training pipeline
#         """
#         self.logger.info('Starting Epoch: %d' % self.opt.start_epoch)
#         self.logger.info('Total Epochs: %d' % self.opt.num_epochs)
#
#         self.epoch = 0
#         for self.epoch in range(self.opt.start_epoch, self.opt.num_epochs):
#             self.run_epoch()
#             self.val()
#
#         # save final model
#         self.save_model("final")
#         self.save_model("best")
#
#
#     def run_epoch(self):
#         """Run a single epoch of training and validation
#         """
#         print("Training")
#         tbar = tqdm(self.train_loader)
#         train_loss = 0.0
#         for batch_idx, inputs in enumerate(tbar):
#             with autograd.record(True):
#                 outputs, losses = self.process_batch(inputs)
#                 mx.nd.waitall()
#
#                 autograd.backward(losses['loss'])
#             self.optimizer.step(self.opt.batch_size, ignore_stale_grad=True)
#
#             train_loss += losses['loss'].asscalar()
#             tbar.set_description('Epoch %d, training loss %.3f' % \
#                                  (self.epoch, train_loss / (batch_idx + 1)))
#
#             if batch_idx % self.opt.log_frequency == 0:
#                 self.logger.info('Epoch %d iteration %04d/%04d: training loss %.3f' %
#                                  (self.epoch, batch_idx, len(self.train_loader),
#                                   train_loss / (batch_idx + 1)))
#             mx.nd.waitall()
#
#


##############################################################################
# You can `Start Training Now`_.
#
# References
# ----------
# .. [Godard17] Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow \
#       "Unsupervised Monocular Depth Estimation with Left-Right Consistency." \
#       Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR). 2017.
#
# .. [Godard19] Clement Godard, Oisin Mac Aodha, Michael Firman and Gabriel Brostow. \
#       "Digging Into Self-Supervised Monocular Depth Estimation." \
#       Proceedings of the IEEE conference on computer vision (ICCV). 2019.
#
