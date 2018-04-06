.. _prepare_datasets:
Prepare Large-Scale Vision Datasets
===================================

Collecting large scale vision datasets is the enabling step for various tasks.
We recognize the tedious manual labors required to prepare datasets with all kinds
of formats and data structures, and providing the simplest scripts for popular Vision
datasets.

We list the preparation scripts which work on different platforms: Linux/Mac OS/Windows, etc.

.. note::
We will make symbolic link of each dataset to user folder ~/.mxnet/datasets/ so that it will
be automatically recognized by `gluonvision` in the future. Making symbolic link won't eat up
your disk space, while being able to ease the filesystem path chaoses.

Pascal VOC
----------
This script will prepare specifed years (07trainvaltest, 2012trainval
and 2012 segmentation augmented set by default) for you.

Link existing dataset on disk stored in ~/Datasets/voc for example:

.. code:: bash

  python examples/datasets/setup_pascal_voc.py --path ~/Datasets/voc

Download the dataset if you don't have it on disk

.. code:: bash

  python examples/datasets/setup_pascal_voc.py --path ~/Datasets/voc --download

Download and overwrite existing tarfiles if files are incomplete or corrupted

.. code:: bash

  python examples/datasets/setup_pascal_voc.py --path ~/Datasets/voc --download --overwrite


.. note:: `path` specifies the `real` path on disk where the extracted files are stored. So make sure you have enough disk space for this operation.


ImageNet
--------
ImageNet, or more specifically ILSVRC2012 dataset is a huge datasets(138G + 6.3G).
You need to download the massive data files by your self from:
http://www.image-net.org/download-images since it use only public for non-commercial use.
An account is required to obtain valid download links.
The targeted files are:

.. list-table::
   :widths: 20 5 30
   :header-rows: 1

   * - File
     - Size
     - SHA-1
   * - ILSVRC2012_img_train.tar
     - 138G
     - 43eda4fe35c1705d6606a6a7a633bc965d194284
   * - ILSVRC2012_img_val.tar
     - 6.3G
     - 5f3f73da3395154b60528b2b2a2caf2374f5f178

If you already have extracted images to *train* and *val* directories, it will be quick job:

.. code:: bash

  python examples/datasets/setup_imagenet.py --path ~/Datasets/ILSVRC2012

If you have downloaded raw tarball files, it's automatically extracted and prepared for you:

.. code:: bash

  python examples/datasets/setup_imagenet.py --path ~/Datasets/ILSVRC2012 --download-dir ~/Downloads/


You can verify the sha-1 checksum, keep in mind that it will be slow for the large training file:

.. code:: bash

  python examples/datasets/setup_imagenet.py --path ~/Datasets/ILSVRC2012 --download-dir ~/Downloads/ --check-sha1


.. note:: `path` specifies the `real` path on disk where the extracted files are stored. So make sure you have enough disk space for this operation.
