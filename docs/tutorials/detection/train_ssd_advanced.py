"""Dive deep into SSD training: 5 tips you may not know
============================================================

In the previous tutorial :ref:`sphx_glr_build_examples_detection_train_ssd_voc.py`, we briefly went through
the fundamental APIs that help building the training pipeline of SSD.

In this article, we will dive deep into the details and introduce something critical
to reproduce SOTA that you may never know by reading the paper and tech reports.

"""

############################################################################
# Loss normalization
# ------------------
# Let's think about the training pipeline.
# You are using mini-batchs of images for training. For each image, the SSD network
# have thousands of predictions to make, either
# "what category does this region belong to?" or
# "how does this anchor box move a little bit to match the real object?".
# Take into consideration the overwhelming number of anchors that are matched
# to ``background`` versus a few ``foreground`` objects (because usually there are a few object on image),
# it is not wise to let the network learn towards such imbalanced targets.


############################################################################
# Initialization
# --------------
# Though SSD networks are based on pre-trained feature extractors, namely ``base_network``
# in the context, there are convolutional layers appended to the ``base_network``
# in order to extend the cascades of feature maps. And there are convolutional
# predictors appened to each output feature map, serve as class predictors and bounding
# box offsets predictors.
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
