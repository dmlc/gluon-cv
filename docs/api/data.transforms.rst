gluoncv.data.transforms
===========================
This file includes various transformations that are critical to vision tasks.

Bounding Box Transforms
----------------------------
.. currentmodule:: gluoncv.data.transforms.bbox

.. autosummary::
    :nosignatures:

    crop

    flip

    resize

    translate

.. currentmodule:: gluoncv.data.transforms

.. autosummary::
    :nosignatures:

    experimental.bbox.random_crop_with_constraints


Image Transforms
---------------------
.. currentmodule:: gluoncv.data.transforms.image

.. autosummary::
    :nosignatures:

    imresize

    resize_long

    resize_short_within

    random_pca_lighting

    random_expand

    random_flip

    resize_contain

    ten_crop


Instance Segmentation Mask Transforms
-------------------------------------
.. currentmodule:: gluoncv.data.transforms.mask

.. autosummary::
    :nosignatures:

    flip

    resize

    to_mask

    fill

Preset Transforms
-----------------
We include presets for reproducing SOTA performances described in
different papers. This is a complimentary section and APIs are prone to changes.

Single Shot Multibox Object Detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluoncv.data.transforms.presets.ssd

.. autosummary::
    :nosignatures:

    load_test

    SSDDefaultTrainTransform

    SSDDefaultValTransform



Faster RCNN
~~~~~~~~~~~

.. currentmodule:: gluoncv.data.transforms.presets.rcnn

.. autosummary::
    :nosignatures:

    load_test

    FasterRCNNDefaultTrainTransform

    FasterRCNNDefaultValTransform

Mask RCNN
~~~~~~~~~

.. currentmodule:: gluoncv.data.transforms.presets.rcnn

.. autosummary::
    :nosignatures:

    load_test

    MaskRCNNDefaultTrainTransform

    MaskRCNNDefaultValTransform


YOLO
~~~~

.. currentmodule:: gluoncv.data.transforms.presets.yolo

.. autosummary::
    :nosignatures:

    load_test

    YOLO3DefaultTrainTransform

    YOLO3DefaultValTransform

API Reference
-------------

.. automodule:: gluoncv.data.transforms.bbox
    :members:
    :imported-members:

.. automodule:: gluoncv.data.transforms.block
    :members:
    :imported-members:

.. automodule:: gluoncv.data.transforms.image
    :members:
    :imported-members:

.. automodule:: gluoncv.data.transforms.mask
    :members:
    :imported-memebers:

.. automodule:: gluoncv.data.transforms.experimental.bbox
    :members:
    :imported-members:

.. automodule:: gluoncv.data.transforms.experimental.image
    :members:
    :imported-members:


.. automodule:: gluoncv.data.transforms.presets.ssd
    :members:
    :imported-members:

.. automodule:: gluoncv.data.transforms.presets.rcnn
    :members:
    :imported-members:

.. automodule:: gluoncv.data.transforms.presets.yolo
    :members:
    :imported-members:
