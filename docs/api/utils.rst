.. role:: hidden
    :class: hidden-section

gluoncv.utils
=================
We implemented a broad range of utility functions which cover visualization, file handler, download and training helpers.

.. currentmodule:: gluoncv.utils
.. automodule:: gluoncv.utils

:hidden:`Visualization`
~~~~~~~~~~~~~~~~~~~~~~~~~

Image Visualization
---------------------
.. autofunction:: gluoncv.utils.viz.plot_image

.. autofunction:: gluoncv.utils.viz.get_color_pallete

Bounding Box Visualization
----------------------------
.. autofunction:: gluoncv.utils.viz.plot_bbox

:hidden:`Miscellaneous`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: gluoncv.utils.download

.. autofunction:: gluoncv.utils.makedirs

.. automodule:: gluoncv.utils.random
    :members:

:hidden:`Training Helpers`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: gluoncv.utils.LR_Scheduler
    :members:

.. autofunction:: gluoncv.utils.set_lr_mult

:hidden:`Bounding Box Utils`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: gluoncv.utils.bbox_iou
