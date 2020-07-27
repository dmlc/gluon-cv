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
#     :width: 80%
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
# class ResnetEncoder(nn.HybridBlock):
#     def __init__(self, backbone, pretrained, num_input_images=1, ctx=cpu(), **kwargs):
#         super(ResnetEncoder, self).__init__()
#
#         self.num_ch_enc = np.array([64, 64, 128, 256, 512])
#
#         resnets = {'resnet18': resnet18_v1b,
#                    'resnet34': resnet34_v1b,
#                    'resnet50': resnet50_v1s,
#                    'resnet101': resnet101_v1s,
#                    'resnet152': resnet152_v1s}
#
#         if backbone not in resnets:
#             raise ValueError("{} is not a valid resnet".format(backbone))
#
#         if num_input_images > 1:
#             pass
#         else:
#             self.encoder = resnets[backbone](pretrained=pretrained, ctx=ctx, **kwargs)
#
#         if backbone not in ('resnet18', 'resnet34'):
#             self.num_ch_enc[1:] *= 4
#
#     def hybrid_forward(self, F, input_image):
#         # pylint: disable=unused-argument, missing-function-docstring
#         self.features = []
#         x = (input_image - 0.45) / 0.225
#         x = self.encoder.conv1(x)
#         x = self.encoder.bn1(x)
#         self.features.append(self.encoder.relu(x))
#         self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
#         self.features.append(self.encoder.layer2(self.features[-1]))
#         self.features.append(self.encoder.layer3(self.features[-1]))
#         self.features.append(self.encoder.layer4(self.features[-1]))
#
#         return self.features
#
#
# The Decoder module is a fully convolutional network with skip architecture, it expolit the featuremaps
# in different scale and concatenating together after upsampling. It is defined as:
#
# class DepthDecoder(nn.HybridBlock):
#     def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1,
#                  use_skips=True):
#         super(DepthDecoder, self).__init__()
#
#         self.num_output_channels = num_output_channels
#         self.use_skips = use_skips
#         self.upsample_mode = 'nearest'
#         self.scales = scales
#
#         self.num_ch_enc = num_ch_enc
#         self.num_ch_dec = np.array([16, 32, 64, 128, 256])
#
#         # decoder
#         with self.name_scope():
#             self.convs = OrderedDict()
#             for i in range(4, -1, -1):
#                 # upconv_0
#                 num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
#                 num_ch_out = self.num_ch_dec[i]
#                 self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
#
#                 # upconv_1
#                 num_ch_in = self.num_ch_dec[i]
#                 if self.use_skips and i > 0:
#                     num_ch_in += self.num_ch_enc[i - 1]
#                 num_ch_out = self.num_ch_dec[i]
#                 self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
#
#             for s in self.scales:
#                 self.convs[("dispconv", s)] = Conv3x3(
#                     self.num_ch_dec[s], self.num_output_channels)
#
#             # register blocks
#             for k in self.convs:
#                 self.register_child(self.convs[k])
#             self.decoder = nn.HybridSequential()
#             self.decoder.add(*list(self.convs.values()))
#
#             self.sigmoid = nn.Activation('sigmoid')
#
#     def hybrid_forward(self, F, input_features):
#         # pylint: disable=unused-argument, missing-function-docstring
#         self.outputs = []
#
#         # decoder
#         x = input_features[-1]
#         for i in range(4, -1, -1):
#             x = self.convs[("upconv", i, 0)](x)
#             x = [F.UpSampling(x, scale=2, sample_type='nearest')]
#             if self.use_skips and i > 0:
#                 x += [input_features[i - 1]]
#             x = F.concat(*x, dim=1)
#             x = self.convs[("upconv", i, 1)](x)
#             if i in self.scales:
#                 self.outputs.append(self.sigmoid(self.convs[("dispconv", i)](x)))
#
#         return self.outputs
#
# Monodepth model is provided in :class:`gluoncv.model_zoo.MonoDepth2`. To get
# Monodepth2 model using ResNet18 base network:
model = gluoncv.model_zoo.get_monodepth2(backbone='resnet18')
print(model)


##############################################################################
# Dataset and Data Augmentation
# -----------------------------
#
##############################################################################
# We provide self-supervised depth estimation datasets in :class:`gluoncv.data`.
# For example, we can easily get the KITTI RAW dataset:
from gluoncv.data.kitti import readlines, dict_batchify_fn

train_filenames = '~/.mxnet/datasets/kitti/splits/eigen_full/train_files.txt'
train_filenames = readlines(train_filenames)
train_dataset = gluoncv.data.KITTIRAWDataset(
    filenames=train_filenames, height=192, width=640,
    frame_idxs=[0], num_scales=4, is_train=True, img_ext='.png')
print('Training images:', len(train_dataset))
# set batch_size = 12 for toy example
batch_size = 12
train_loader = gluon.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, batchify_fn=dict_batchify_fn,
    num_workers=12, pin_memory=True, last_batch='discard')
##############################################################################
# You can `Start Training Now`_.
#
# References
# ----------
# .. [Godard19] Clement Godard, Oisin Mac Aodha, Michael Firman, Gabriel Brostow. \
#       "Digging Into Self-Supervised Monocular Depth Estimation." \
#       Proceedings of the IEEE conference on computer vision. 2019.
#
