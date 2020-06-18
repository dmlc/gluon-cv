"""12. Run an object detection model on NVIDIA Jetson module
==========================================================

This tutorial shows how to install MXNet v1.6 (with Jetson support) and GluonCV on a Jetson module and deploy a pre-trained GluonCV model for object detection.

What's in this tutorial?
------------------------
This tutorial shows how to:

1. Install MXNet v1.6 along with its dependencies on a Jetson module

2. Install GluonCV and its dependencies on the module

3. Deploy a pre-trained YOLO model for object detection on the module

.. note::
    This tutorial has been tested on Jetson Xavier AGX and Jetson TX2 modules.

Who's this tutorial for?
------------------------
This tutorial would benefit developers working on Jetson modules implementing deep learning applications. It assumes that readers have a Jetson module setup with Jetpack installed, are familiar with the Jetson working environment and are somewhat familiar with deep learning using MXNet.

Prerequisites
-------------
To complete this tutorial, you need:

* A `Jetson module <https://developer.nvidia.com/embedded/develop/hardware>`__ setup with `Jetpack 4.4 <https://docs.nvidia.com/jetson/jetpack/release-notes/>`__ installed using NVIDIA `SDK Manager <https://developer.nvidia.com/nvidia-sdk-manager>`__

* Display (needed to view matplotlib plot) and keyboard setup to directly open shell on the module

* `Swapfile <https://help.ubuntu.com/community/SwapFaq>`__ installed, especially on Jetson Nano for additional memory (increase memory if the inference script terminates with a `Killed` message)

Installing MXNet v1.6 with Jetson support
-----------------------------------------
To install MXNet with Jetson support, you can follow the `installation guide <https://mxnet.apache.org/get_started/jetson_setup>`__ on MXNet official website.

Alternatively, you can also directly install MXNet v1.6 wheel with Jetson support, hosted on a public s3 bucket. Here are the steps to install this wheel:

.. note::
    WARNING: this MXNet wheel is provided for your convenience but it contains packages that are not provided nor endorsed by the Apache Software Foundation.
    As such, they might contain software components with more restrictive licenses than the Apache License and you'll need to decide whether they are appropriate for your usage. Like all Apache Releases, the
    official Apache MXNet (incubating) releases consist of source code only and are found `here <https://mxnet.apache.org/get_started/download>`__ .

We start by installing MXNet dependencies

.. code-block:: bash

    sudo apt-get update
    sudo apt-get install -y git build-essential libopenblas-dev python3-pip
    sudo pip3 install -U pip

Then we download and install MXNet v1.6 wheel with Jetson support

.. code-block:: bash

    wget https://mxnet-public.s3.us-east-2.amazonaws.com/install/jetson/1.6.0/mxnet_cu102-1.6.0-py2.py3-none-linux_aarch64.whl
    sudo pip3 install mxnet_cu102-1.6.0-py2.py3-none-linux_aarch64.whl

And we are done. You can test the installation now by importing mxnet from python3

.. code-block:: bash

    >>> python3 -c 'import mxnet'

Installing GluonCV and its dependencies
---------------------------------------
We can install GluonCV on Jetson module using the following commands:

.. code-block:: bash

    sudo apt-get update
    sudo apt-get install -y python3-scipy python3-pil python3-matplotlib
    sudo apt autoremove -y
    sudo pip3 install gluoncv

Running a pre-trained GluonCV YOLOv3 model on Jetson
----------------------------------------------------
We are now ready to deploy a pre-trained model and run inference on a Jetson module. In this tutorial we are using YOLOv3 model trained on Pascal VOC dataset with Darknet53 as the base model. The object detection script below can be run with either cpu/gpu context using python3.

.. note::

    If running with GPU context, set the environment variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable cuDNN autotune

    .. code-block:: bash

        export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

Here's the object detection python script:

.. code-block:: python

    from gluoncv import model_zoo, data, utils
    from matplotlib import pyplot as plt
    import mxnet as mx

    # set context
    ctx = mx.gpu()

    # load model
    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx=ctx)

    # load input image
    im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
                            'mxnet-ssd/master/data/demo/dog.jpg',
                            path='dog.jpg')
    x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
    x = x.as_in_context(ctx)

    # call forward and show plot
    class_IDs, scores, bounding_boxs = net(x)
    ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                            class_IDs[0], class_names=net.classes)
    plt.show()

This is the input image:

.. image:: https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg

After running the above script, you should get the following plot as output:

.. image:: https://gluon-cv.mxnet.io/_images/sphx_glr_demo_yolo_001.png


"""
