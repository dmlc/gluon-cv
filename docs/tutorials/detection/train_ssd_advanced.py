"""Dive deep into SSD training: 5 tips you may not know
============================================================

In the previous tutorial :ref:`sphx_glr_build_examples_detection_train_ssd_voc.py`, we briefly went through
the fundamental APIs that help building the training pipeline of SSD.

In this article, we will dive deep into the details and introduce something critical
to reproduce SOTA that you may never know by reading the paper and tech reports.

.. contents:: :local:

"""

############################################################################
# Loss normalization: use batch-wise norm instead of sample-wise norm
# -------------------------------------------------------------------
# The training objective mentioned in paper is a weighted summation of localization
# loss(loc) and the confidence loss(conf).
#
# .. math:: L(x, c, l, g) = \frac{1}{N} (L_{conf}(x, c) + \alpha L_{loc}(x, l, g))
#
# But the question is, what is the proper way to calculate ``N``? Should we sum up
# ``N`` across the entire batch, or use per-sample ``N`` instead?
#
# let's use some fake data to illustrate

import mxnet as mx
x = mx.random.uniform(shape=(2, 3, 300, 300))  # use batch-size 2
# suppose image 1 has single object
id1 = mx.nd.array([1])
bbox1 = mx.nd.array([[10, 20, 80, 90]])  # xmin, ymin, xmax, ymax
# suppose image 2 has 4 objects
id2 = mx.nd.array([1, 3, 5, 7])
bbox2 = mx.nd.array([[10, 10, 30, 30], [40, 40, 60, 60], [50, 50, 90, 90], [100, 110, 120, 140]])

############################################################################
# put them together into batch, by padding some sentinal values -1
gt_ids = mx.nd.ones(shape=(2, 4)) * -1
gt_ids[0, :1] = id1
gt_ids[1, :4] = id2
print('class_ids:', gt_ids)

############################################################################
gt_boxes = mx.nd.ones(shape=(2, 4, 4)) * -1
gt_boxes[0, :1, :] = bbox1
gt_boxes[1, :, :] = bbox2
print('bounding boxes:', gt_boxes)

############################################################################
# We use a vgg16 atrous 300x300 SSD model in this example. For demo purpose, we
# don't use any pretrained weights here
from gluonvision import model_zoo
net = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained_base=False, pretrained=False)

############################################################################
# Some preparation before training
from mxnet import gluon
net.initialize()
conf_loss = gluon.loss.SoftmaxCrossEntropyLoss()
loc_loss = gluon.loss.HuberLoss()

############################################################################
# Simulate the training steps:
from mxnet import autograd
with autograd.record():
    # 1. forward pass
    cls_preds, box_preds, anchors = net(x)
    # 2. generate training targets
    cls_targets, box_targets, box_masks = net.target_generator(
        anchors, cls_preds, gt_boxes, gt_ids)
    num_positive = (cls_targets > 0).sum().asscalar()
    cls_mask = (cls_targets >= 0).expand_dims(axis=-1)  # negative targets should be ignored in loss
    # 3 losses, here we have two options, batch-wise or sample wise norm
    # 3.1 batch wise normalization: divide loss by the summation of num positive targets in batch
    batch_conf_loss = conf_loss(cls_preds, cls_targets, cls_mask) / num_positive
    batch_loc_loss = loc_loss(box_preds, box_targets, box_masks) / num_positive
    # 3.2 sample wise normalization: divide by num positive targets in this sample(image)
    sample_num_positive = (cls_targets > 0).sum(axis=0, exclude=True)
    sample_conf_loss = conf_loss(cls_preds, cls_targets, cls_mask) / sample_num_positive
    sample_loc_loss = loc_loss(box_preds, box_targets, box_masks) / sample_num_positive
    # Since ``conf_loss`` and ``loc_loss`` calculate the mean of such loss, we want
    # to rescale it back to loss per image.
    rescale_conf = cls_preds.size / cls_preds.shape[0]
    rescale_loc = box_preds.size / box_preds.shape[0]
    # then call backward and step, to update the weights, etc..
    # L = conf_loss + loc_loss * alpha
    # L.backward()

############################################################################
# The norms are different, but sample wise norms sum up to the total
print('batch-wise num_positive:', num_positive)
print('sample-wise num_positive:', sample_num_positive)

############################################################################
# .. note::
#     The per image ``num_positive`` is no longer 1 and 4 because multiple anchor
#     boxes can be matched to a single object

############################################################################
# Compare the losses
print('batch-wise norm conf loss:', batch_conf_loss * rescale_conf)
print('sample-wise norm conf loss:', sample_conf_loss * rescale_conf)

############################################################################
print('batch-wise norm loc loss:', batch_loc_loss * rescale_loc)
print('sample-wise norm loc loss:', sample_loc_loss * rescale_loc)

############################################################################
# Which one is better?
# At first glance, it is complicated to say which one is theoretically better
# because batch-wise norm ensures loss is well normalized by global statistics
# while sample-wise norm ensures gradients won't explode in some extreame cases where
# there are seveal hundreds of objects in a single image.
# In such case it would cause other samples in the same
# batch being suppressed by this unusually large norm.
#
# In our experiments, batch-wise norm is always better on Pascal VOC dataset,
# contributing 1~2% mAP gain. However, you would definitely try comparing them
# when you got a new dataset and probably new model.


############################################################################
# Initializer matters: try
# --------------------------------------------------------
# Though SSD networks are based on pre-trained feature extractors, namely ``base_network``
# in the context, there are convolutional layers appended to the ``base_network``
# in order to extend the cascades of feature maps.
#
# And there are convolutional
# predictors appened to each output feature map, serve as class predictors and bounding
# box offsets predictors.
#
# For these added layers, we will randomly initialize them before training.
from gluonvision import model_zoo
import mxnet as mx
# don't load pretrained for this demo
net = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained=False, pretrained_base=False)
# random init
net.initialize()
# gluon only infer shape when real input data is used
net(mx.nd.zeros(shape=(1, 3, 300, 300)))
# now we have real shape for each parameter
predictors = [(k, v) for k, v in net.collect_params().items() if 'predictor' in k]
name, pred = predictors[0]
print(name, pred)

############################################################################
# we can initialize it using different initializer, such as ``Normal``, ``Xavier``.
pred.initialize(mx.init.Uniform(), force_reinit=True)
print('param shape:', pred.data().shape, 'peek first 20 elem:', pred.data().reshape((-1))[:20])

############################################################################
# Simply switching from ``Uniform`` to ``Xavier`` can get ~1% mAP gain after full training.
pred.initialize(mx.init.Xavier(rnd_type='gaussian', magnitude=2, factor_type='out'), force_reinit=True)
print('param shape:', pred.data().shape, 'peek first 20 elem:', pred.data().reshape((-1))[:20])


############################################################################
# Interprete confidence scores: process each class separately
# -----------------------------------------------------------
# If we revisit the per-class confidence predictions, its shape is (``B``, ``A``, ``N+1``)
# where ``B`` is the batch size, ``A`` is the number of anchor boxes,
# ``N`` is the number of foreground classes.
print('class prediction shape:', cls_preds.shape)

############################################################################
# There are basically two ways to handle the prediction:
# 1. take argmax of the prediction along class axis, i.e., the most likely class this object belongs to
# 2. process ``N`` foreground classes separately, in this case, rank 2 class for example still have a
# chance to survive as the final prediction.
#
# Consider such a case
cls_pred = mx.nd.array([-1, -2, 3, 4, 6.5, 6.4])
cls_prob = mx.nd.softmax(cls_pred, axis=-1)
for k, v in zip(['bg', 'apple', 'orange', 'person', 'dog', 'cat'], cls_prob.asnumpy().tolist()):
    print(k, v)

############################################################################
# The probabilities of the dog and cat are so close, if we use method 1,
# we are losing the bet when cat is the correct decision.
#
# It turns out that using method2, we achieved 0.5~0.8 better mAP in evaluation.
#
# One drawback of method 2 is that is significantly slower than method 1.
# In terms of N classes, it's O(N) versus O(1) in method 1.
#
# .. hint::
#   You can checkout :doc:`gluonvision.model_zoo.coders.MultiClassDecoder` and
#   :doc:`gluonvision.model_zoo.coders.MultiPerClassDecoder`, which implements method 1 and 2, respectively.
