"""05. Deep dive into SSD training: 3 tips to boost performance
===============================================================

In the previous tutorial :ref:`sphx_glr_build_examples_detection_train_ssd_voc.py`,
we briefly went through the basic APIs that help building the training pipeline of SSD.

In this article, we will dive deep into the details and introduce tricks that
important for reproducing state-of-the-art performance.
These are the hidden pitfalls that are usually missing in papers and tech reports.

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
# To illustrate this, please generate some dummy data:

import mxnet as mx
x = mx.random.uniform(shape=(2, 3, 300, 300))  # use batch-size 2
# suppose image 1 has single object
id1 = mx.nd.array([1])
bbox1 = mx.nd.array([[10, 20, 80, 90]])  # xmin, ymin, xmax, ymax
# suppose image 2 has 4 objects
id2 = mx.nd.array([1, 3, 5, 7])
bbox2 = mx.nd.array([[10, 10, 30, 30], [40, 40, 60, 60], [50, 50, 90, 90], [100, 110, 120, 140]])

############################################################################
# Then, combine them into a batch by padding -1 as sentinal values:
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
from gluoncv import model_zoo
net = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained_base=False, pretrained=False)

############################################################################
# Some preparation before training
from mxnet import gluon
net.initialize()
conf_loss = gluon.loss.SoftmaxCrossEntropyLoss()
loc_loss = gluon.loss.HuberLoss()

############################################################################
# Simulate the training steps by manually compute losses:
# You can always use ``gluoncv.loss.SSDMultiBoxLoss`` which fulfills this function.
from mxnet import autograd
from gluoncv.model_zoo.ssd.target import SSDTargetGenerator
target_generator = SSDTargetGenerator()
with autograd.record():
    # 1. forward pass
    cls_preds, box_preds, anchors = net(x)
    # 2. generate training targets
    cls_targets, box_targets, box_masks = target_generator(
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
# The norms are different, but sample-wise norms sum up to be the same with
# batch-wise norm
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
# At first glance, it is hard to say which one is theoretically better
# because batch-wise norm ensures loss is well normalized by global statistics
# while sample-wise norm ensures gradients won't explode in some extreme cases where
# there are hundreds of objects in a single image.
# In such case it would cause other samples in the same
# batch to be suppressed by this unusually large norm.
#
# In our experiments, batch-wise norm is always better on Pascal VOC dataset,
# contributing 1~2% mAP gain. However, you should definitely try both of them
# when you use a new dataset or a new model.


############################################################################
# Initializer matters: don't stick to one single initializer
# --------------------------------------------------------
# While SSD networks are based on pre-trained feature extractors (called the ``base_network``),
# we also append uninitialized convolutional layers to the ``base_network``
# to extend the cascades of feature maps.
#
# There are also convolutional
# predictors appended to each output feature map, serving as class predictors and bounding
# box offsets predictors.
#
# For these added layers, we must initialize them before training.
from gluoncv import model_zoo
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
# we can initialize it with different initializers, such as ``Normal`` or ``Xavier``.
pred.initialize(mx.init.Uniform(), force_reinit=True)
print('param shape:', pred.data().shape, 'peek first 20 elem:', pred.data().reshape((-1))[:20])

############################################################################
# Simply switching from ``Uniform`` to ``Xavier`` can produce ~1% mAP gain.
pred.initialize(mx.init.Xavier(rnd_type='gaussian', magnitude=2, factor_type='out'), force_reinit=True)
print('param shape:', pred.data().shape, 'peek first 20 elem:', pred.data().reshape((-1))[:20])


############################################################################
# Interpreting confidence scores: process each class separately
# -----------------------------------------------------------
# If we revisit the per-class confidence predictions, its shape is (``B``, ``A``, ``N+1``),
# where ``B`` is the batch size, ``A`` is the number of anchor boxes,
# ``N`` is the number of foreground classes.
print('class prediction shape:', cls_preds.shape)

############################################################################
# There are two ways we can handle the prediction:
#
# 1. take argmax of the prediction along the class axis. This way, only the
# the most probable class is considered.
#
# 2. process ``N`` foreground classes separately. This way, the second most
# probable class, for example, still has a
# chance of surviving as the final prediction.
#
# Consider this example:
cls_pred = mx.nd.array([-1, -2, 3, 4, 6.5, 6.4])
cls_prob = mx.nd.softmax(cls_pred, axis=-1)
for k, v in zip(['bg', 'apple', 'orange', 'person', 'dog', 'cat'], cls_prob.asnumpy().tolist()):
    print(k, v)

############################################################################
# The probabilities of dog and cat are so close that if we use method 1,
# we are quite likely to lose the bet when cat is the correct decision.
#
# It turns out that by switching from method 1 to method 2, we gain 0.5~0.8 mAP in evaluation.
#
# One obvious drawback of method 2 is that it is significantly slower than method 1.
# For N classes, method 2 has O(N) complexity while method 1 is always O(1).
# This may or may not be a problem depending on the use case, but feel free to switch between them if you want.
#
# .. hint::
#   Checkout :py:meth:`gluoncv.nn.coder.MultiClassDecoder` and
#   :py:meth:`gluoncv.nn.coder.MultiPerClassDecoder` for implementations of method 1 and 2, respectively.
