.. role:: hidden
    :class: hidden-section

Utility Functions
=================
We implemented a broad range of utility functions which cover visualization, file handler, download and training helpers.

:hidden:`Visualization`
~~~~~~~~~~~~~~~~~~~~~~~~~

Image Visualization
---------------------
.. autofunction:: gluonvision.utils.viz.plot_image

Bounding Box Visualization
----------------------------
.. autofunction:: gluonvision.utils.viz.plot_bbox

:hidden:`Miscellaneous`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: gluonvision.utils.download

.. autofunction:: gluonvision.utils.makedirs

.. automodule:: gluonvision.utils.random
    :members:

:hidden:`Training Helpers`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: gluonvision.utils.PolyLRScheduler
    :members:

.. autofunction:: gluonvision.utils.set_lr_mult

:hidden:`Bouding Box Utils`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: gluonvision.utils.bbox_iou
