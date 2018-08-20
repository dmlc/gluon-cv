"""2. Train Mask RCNN end-to-end on MS COCO
===========================================

This tutorial goes through the steps for training a Mask R-CNN [He17]_ instance segmentation model
provided by GluonCV.

Mask R-CNN is an extension to the Faster R-CNN [Ren15]_ object detection model.
As such, this tutorial is also an extension to :doc:`../examples_detection/train_faster_rcnn_voc`.
We will focus on the extra work on top of Faster R-CNN to show how to use GluonCV components
to construct a Mask R-CNN model.

It is highly recommended to read the original papers [Girshick14]_, [Girshick15]_, [Ren15]_, [He17]_
to learn more about the ideas behind Mask R-CNN.
Appendix from [He16]_ and experiment detail from [Lin17]_ may also be useful reference.

.. hint::

    Please first go through this :ref:`sphx_glr_build_examples_datasets_mscoco.py` tutorial to
    setup MSCOCO dataset on your disk.

.. hint::

    You can skip the rest of this tutorial and start training your Mask RCNN model
    right away by downloading this script:

    :download:`Download train_mask_rcnn.py<../../../scripts/instance/mask_rcnn/train_mask_rcnn.py>`

    Example usage:

    Train a default resnet50_v1b model with COCO dataset on GPU 0:

    .. code-block:: bash

        python train_mask_rcnn.py --gpus 0

    Train on GPU 0,1,2,3:

    .. code-block:: bash

        python train_mask_rcnn.py --gpus 0,1,2,3

    Check the supported arguments:

    .. code-block:: bash

        python train_mask_rcnn.py --help

"""

##########################################################
# Dataset
# -------
#
# Make sure COCO dataset has been set up on your disk.
# Then, we are ready to load training and validation images.

from gluoncv.data import COCOInstance
# typically we use train2017 (i.e. train2014 + minival35k) split as training data
# COCO dataset actually has images without any objects annotated,
# which must be skipped during training to prevent empty labels
train_dataset = COCOInstance(splits='instances_train2017', skip_empty=True)
# and val2014 (i.e. minival5k) test as validation data
val_dataset = COCOInstance(splits='instances_val2017', skip_empty=False)

print('Training images:', len(train_dataset))
print('Validation images:', len(val_dataset))

##########################################################
# Data transform
# --------------
# We can read an (image, label, segm) tuple from the training dataset:
train_image, train_label, train_segm = train_dataset[6]
bboxes = train_label[:, :4]
cids = train_label[:, 4:5]
print('image:', train_image.shape)
print('bboxes:', bboxes.shape, 'class ids:', cids.shape)
# segm is a list of polygons which are arrays of points on the object boundary
print('masks', [[poly.shape for poly in polys] for polys in train_segm])

##############################################################################
# Plot the image with boxes and labels:
from matplotlib import pyplot as plt
from gluoncv.utils import viz

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax = viz.plot_bbox(train_image, bboxes, labels=cids, class_names=train_dataset.classes, ax=ax)
plt.show()

##############################################################################
# To actually see the object segmentation, we need to convert polygons to masks
import numpy as np
from gluoncv.data.transforms import mask as tmask
width, height = train_image.shape[1], train_image.shape[0]
train_masks = np.stack([tmask.to_mask(polys, (width, height)) for polys in train_segm])
plt_image = viz.plot_mask(train_image, train_masks)

##############################################################################
# Now plot the image with boxes, labels and masks
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax = viz.plot_bbox(plt_image, bboxes, labels=cids, class_names=train_dataset.classes, ax=ax)
plt.show()

##############################################################################
# Data transforms, i.e. decoding and transformation, are identical to Faster R-CNN
# with the exception of segmentation polygons as an additional input.
# :py:class:`gluoncv.data.transforms.presets.rcnn.MaskRCNNDefaultTrainTransform`
# converts the segmentation polygons to binary segmentation mask.
# :py:class:`gluoncv.data.transforms.presets.rcnn.MaskRCNNDefaultValTransform`
# ignores the segmentation polygons and returns image tensor and ``[im_height, im_width, im_scale]``.
from gluoncv.data.transforms import presets
from gluoncv import utils
from mxnet import nd

##############################################################################
short, max_size = 600, 1000  # resize image to short side 600 px, but keep maximum length within 1000
train_transform = presets.rcnn.MaskRCNNDefaultTrainTransform(short, max_size)
val_transform = presets.rcnn.MaskRCNNDefaultValTransform(short, max_size)

##############################################################################
utils.random.seed(233)  # fix seed in this tutorial

##############################################################################
# apply transforms to train image
train_image2, train_label2, train_masks2 = train_transform(train_image, train_label, train_segm)
print('tensor shape:', train_image2.shape)
print('box and id shape:', train_label2.shape)
print('mask shape', train_masks2.shape)

##############################################################################
# Images in tensor are distorted because they no longer sit in (0, 255) range.
# Let's convert them back so we can see them clearly.
plt_image2 = train_image2.transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
plt_image2 = (plt_image2 * 255).asnumpy().astype('uint8')

##############################################################################
# The transform already converted polygons to masks and we plot them directly.
width, height = plt_image2.shape[1], plt_image2.shape[0]
plt_image2 = viz.plot_mask(plt_image2, train_masks2)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax = viz.plot_bbox(plt_image2, train_label2[:, :4],
                   labels=train_label2[:, 4:5],
                   class_names=train_dataset.classes,
                   ax=ax)
plt.show()

##########################################################
# Data Loader
# -----------
# Data loader is identical to Faster R-CNN with the difference of mask input and output.

from gluoncv.data.batchify import Tuple, Append
from mxnet.gluon.data import DataLoader

batch_size = 2  # for tutorial, we use smaller batch-size
num_workers = 0  # you can make it larger(if your CPU has more cores) to accelerate data loading

train_bfn = Tuple(*[Append() for _ in range(3)])
train_loader = DataLoader(train_dataset.transform(train_transform), batch_size, shuffle=True,
                          batchify_fn=train_bfn, last_batch='rollover', num_workers=num_workers)
val_bfn = Tuple(*[Append() for _ in range(2)])
val_loader = DataLoader(val_dataset.transform(val_transform), batch_size, shuffle=False,
                        batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)

for ib, batch in enumerate(train_loader):
    if ib > 3:
        break
    print('data 0:', batch[0][0].shape, 'label 0:', batch[1][0].shape, 'mask 0:', batch[2][0].shape)
    print('data 1:', batch[0][1].shape, 'label 1:', batch[1][1].shape, 'mask 1:', batch[2][1].shape)

##########################################################
# Mask RCNN Network
# -------------------
# In GluonCV, Mask RCNN network :py:class:`gluoncv.model_zoo.MaskRCNN`
# is inherited from Faster RCNN network :py:class:`gluoncv.model_zoo.FasterRCNN`.
#
# `Gluon Model Zoo <../../model_zoo/index.html>`__ has some Mask RCNN pretrained networks.
# You can load your favorate one with one simple line of code:
#
# .. hint::
#
#    To avoid downloading mdoel in this tutorial, we set ``pretrained_base=False``,
#    in practice we usually want to load pre-trained imagenet models by setting
#    ``pretrained_base=True``.
from gluoncv import model_zoo
net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained_base=False)
print(net)

##############################################################################
# Mask-RCNN has identical inputs but produces an additional output.
# ``cids`` are the class labels,
# ``scores`` are confidence scores of each prediction,
# ``bboxes`` are absolute coordinates of corresponding bounding boxes.
# ``masks`` are predicted segmentation masks corresponding to each bounding box
import mxnet as mx
x = mx.nd.zeros(shape=(1, 3, 600, 800))
net.initialize()
cids, scores, bboxes, masks = net(x)

##############################################################################
# During training, an additional output is returned:
# ``mask_preds`` are per class masks predictions
# in addition to ``cls_preds``, ``box_preds``.
from mxnet import autograd
with autograd.train_mode():
    # this time we need ground-truth to generate high quality roi proposals during training
    gt_box = mx.nd.zeros(shape=(1, 1, 4))
    cls_preds, box_preds, mask_preds, roi, samples, matches, rpn_score, rpn_box, anchors = net(x, gt_box)

##########################################################
# Training losses
# ----------------
# There are one additional losses in Mask-RCNN.

# the loss to penalize incorrect foreground/background prediction
rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
# the loss to penalize inaccurate anchor boxes
rpn_box_loss = mx.gluon.loss.HuberLoss(rho=1/9.)  # == smoothl1
# the loss to penalize incorrect classification prediction.
rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
# and finally the loss to penalize inaccurate proposals
rcnn_box_loss = mx.gluon.loss.HuberLoss()  # == smoothl1
# the loss to penalize incorrect segmentation pixel prediction
rcnn_mask_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)

##########################################################
# Training targets
# ----------------
# RPN and RCNN training target are the same as in :doc:`../examples_detection/train_faster_rcnn_voc`.

##############################################################################
# We also push RPN targets computation to CPU workers, so network is passed to transforms
train_transform = presets.rcnn.MaskRCNNDefaultTrainTransform(short, max_size, net)
# return images, labels, masks, rpn_cls_targets, rpn_box_targets, rpn_box_masks loosely
batchify_fn = Tuple(*[Append() for _ in range(6)])
# For the next part, we only use batch size 1
batch_size = 1
train_loader = DataLoader(train_dataset.transform(train_transform), batch_size, shuffle=True,
                          batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)

##########################################################
# Mask targets are generated with the intermediate outputs after rcnn target is generated.

for ib, batch in enumerate(train_loader):
    if ib > 0:
        break
    with autograd.train_mode():
        for data, label, masks, rpn_cls_targets, rpn_box_targets, rpn_box_masks in zip(*batch):
            gt_label = label[:, :, 4:5]
            gt_box = label[:, :, :4]
            # network forward
            cls_preds, box_preds, mask_preds, roi, samples, matches, rpn_score, rpn_box, anchors = net(data, gt_box)
            # generate targets for rcnn
            cls_targets, box_targets, box_masks = net.target_generator(roi, samples, matches, gt_label, gt_box)
            # generate targets for mask head
            mask_targets, mask_masks = net.mask_target(roi, masks, matches, cls_targets)
            print('data:', data.shape)
            # box and class labels
            print('box:', gt_box.shape)
            print('label:', gt_label.shape)
            # -1 marks ignored label
            print('rpn cls label:', rpn_cls_targets.shape)
            # mask out ignored box label
            print('rpn box label:', rpn_box_targets.shape)
            print('rpn box mask:', rpn_box_masks.shape)
            # rcnn does not have ignored label
            print('rcnn cls label:', cls_targets.shape)
            # mask out ignored box label
            print('rcnn box label:', box_targets.shape)
            print('rcnn box mask:', box_masks.shape)
            print('rcnn mask label:', mask_targets.shape)
            print('rcnn mask mask:', mask_masks.shape)

##########################################################
# Training loop
# -------------
# After we have defined loss function and generated training targets, we can write the training loop.

for ib, batch in enumerate(train_loader):
    if ib > 0:
        break
    with autograd.record():
        for data, label, masks, rpn_cls_targets, rpn_box_targets, rpn_box_masks in zip(*batch):
            gt_label = label[:, :, 4:5]
            gt_box = label[:, :, :4]
            # network forward
            cls_preds, box_preds, mask_preds, roi, samples, matches, rpn_score, rpn_box, anchors = net(data, gt_box)
            # generate targets for rcnn
            cls_targets, box_targets, box_masks = net.target_generator(roi, samples, matches, gt_label, gt_box)
            # generate targets for mask head
            mask_targets, mask_masks = net.mask_target(roi, masks, matches, cls_targets)

            # losses of rpn
            rpn_score = rpn_score.squeeze(axis=-1)
            num_rpn_pos = (rpn_cls_targets >= 0).sum()
            rpn_loss1 = rpn_cls_loss(rpn_score, rpn_cls_targets, rpn_cls_targets >= 0) * rpn_cls_targets.size / num_rpn_pos
            rpn_loss2 = rpn_box_loss(rpn_box, rpn_box_targets, rpn_box_masks) * rpn_box.size / num_rpn_pos

            # losses of rcnn
            num_rcnn_pos = (cls_targets >= 0).sum()
            rcnn_loss1 = rcnn_cls_loss(cls_preds, cls_targets, cls_targets >= 0) * cls_targets.size / cls_targets.shape[0] / num_rcnn_pos
            rcnn_loss2 = rcnn_box_loss(box_preds, box_targets, box_masks) * box_preds.size / box_preds.shape[0] / num_rcnn_pos

            # loss of mask
            mask_loss = rcnn_mask_loss(mask_preds, mask_targets, mask_masks) * mask_targets.size / mask_targets.shape[0] / mask_masks.sum()

        # some standard gluon training steps:
        # autograd.backward([rpn_loss1, rpn_loss2, rcnn_loss1, rcnn_loss2, mask_loss])
        # trainer.step(batch_size)

##############################################################################
# .. hint::
#
#   Please checkout the full :download:`training script <../../../scripts/instance/mask_rcnn/train_mask_rcnn.py>` for complete implementation.

##########################################################
# References
# ----------
#
# .. [Girshick14] Ross Girshick and Jeff Donahue and Trevor Darrell and Jitendra Malik. Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation. CVPR 2014.
# .. [Girshick15] Ross Girshick. Fast {R-CNN}. ICCV 2015.
# .. [Ren15] Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun. Faster {R-CNN}: Towards Real-Time Object Detection with Region Proposal Networks. NIPS 2015.
# .. [He16] Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun. Deep Residual Learning for Image Recognition. CVPR 2016.
# .. [Lin17] Tsung-Yi Lin and Piotr Dollár and Ross Girshick and Kaiming He and Bharath Hariharan and Serge Belongie. Feature Pyramid Networks for Object Detection. CVPR 2017.
# .. [He17] Kaiming He and Georgia Gkioxari and Piotr Dollár and and Ross Girshick. Mask {R-CNN}. ICCV 2017.
