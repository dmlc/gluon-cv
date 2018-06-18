gluoncv.nn
==========

Neural Network Components.

.. hint::

  Not every component listed here is `HybridBlock <https://mxnet.incubator.apache.org/tutorials/gluon/hybrid.html>`_,
  which means some of them are not hybridizable.
  However, we are trying our best to make sure components required during inference are hybridizable
  so the entire network can be exported and run in other languages.

  For example, encoders are usually non-hybridizable but are only required during training.
  In contrast, decoders are mostly `HybridBlock`s.

Bounding Box
------------

Blocks that apply bounding box related functions.

.. currentmodule:: gluoncv.nn.bbox

.. autosummary::
    :nosignatures:

    BBoxCornerToCenter

    BBoxCenterToCorner

    BBoxSplit

    BBoxArea

Coders
------

Encoders are used to encode training targets before we apply loss functions.
Decoders are used to restore predicted values by inverting the operations done in encoders.
They often come as a pair in order to make the results consistent.

.. currentmodule:: gluoncv.nn.coder

.. autosummary::
    :nosignatures:

    NormalizedBoxCenterEncoder

    NormalizedBoxCenterDecoder

    MultiClassEncoder

    MultiClassDecoder

    MultiPerClassDecoder

    SigmoidClassEncoder


Feature
--------

Feature layers are components that either extract partial networks as feature extractor or extend
them with new layers.

.. currentmodule:: gluoncv.nn.feature

.. autosummary::
    :nosignatures:

    FeatureExtractor

    FeatureExpander



Matchers
--------

Matchers are often used by object detection tasks whose target is to find the matchings between
anchor boxes(very popular in object detection) and ground truths.

.. currentmodule:: gluoncv.nn.matcher

.. autosummary::
    :nosignatures:

    CompositeMatcher

    BipartiteMatcher

    MaximumMatcher

Predictors
----------

Predictors are common neural network components which are specifically used to predict values.
Depending on the purpose, it may vary from Convolution or Fully Connected.

.. currentmodule:: gluoncv.nn.predictor

.. autosummary::
    :nosignatures:

    ConvPredictor

    FCPredictor

Samplers
--------

Samples are often used after matching layers which is to determine positive/negative/ignored samples.

For example, a NaiveSampler simply returns all matched samples as positive, and all un-matched samples as negative.

This behavior is sometimes prone to vulnerability because training objective is not balanced.
Please see `OHEMSampler` and `QuotaSampler` for more advanced sampling strategies.

.. currentmodule:: gluoncv.nn.sampler

.. autosummary::
    :nosignatures:

    NaiveSampler

    OHEMSampler

    QuotaSampler


API Reference
-------------

.. automodule:: gluoncv.nn.bbox
    :members:

.. automodule:: gluoncv.nn.coder
    :members:

.. automodule:: gluoncv.nn.feature
    :members:

.. automodule:: gluoncv.nn.predictor
    :members:

.. automodule:: gluoncv.nn.matcher
    :members:

.. automodule:: gluoncv.nn.sampler
    :members:
