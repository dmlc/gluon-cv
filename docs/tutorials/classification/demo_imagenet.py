"""Getting Started with ImageNet Pre-trained Models
===================================================

`ImageNet <http://www.image-net.org/>`__ is a
large labeled dataset of real-world images. It is the most
well-known dataset for computer vision tasks.

|imagenet|

In this tutorial, we will demonstrate how a well-trained model
classifies real life images.

Specifically, we offer a script to load a pretrained ``ResNet50_v2`` model.
For a list of models we have, please visit our `Model Zoo <../model_zoo/index.html>`__.

Different from ``CIFAR10``, training a model on ImageNet is much more difficult.
We will have more discussions on ImageNet in other tutorials.

Demo and Benchmark
------------------

A model trained on ImageNet can classify images into 1000 classes, this makes it
much more powerful than the one we showed in `CIFAR10 demo tutorial <demo_cifar10.html>`__.

:download:`Download Python Script demo_imagenet.py<../../../scripts/classification/imagenet/demo_imagenet.py>`

With this script, you can load a pre-trained model and predict on any image you have.

Let's take the photo of Mt. Baker again as the first photo.

|image0|

We can make prediction by

::

    python demo_imagenet.py --model ResNet50_v2 --input-pic mt_baker.jpg

And the model thinks that

::

    The input picture is classified to be
    	[volcano], with probability 0.558.
    	[alp], with probability 0.398.
    	[valley], with probability 0.018.
    	[lakeside], with probability 0.006.
    	[mountain_tent], with probability 0.006.

This time it does a perfect job. Note that we have listed the top five
possible classes, because with 1000 classes the model may not always rate the
correct answer with the highest rank. Besides the top-1 accuracy, we also
consider top-5 accuracy as a measurement of how well a model can predict.

Next Step
---------

If you would like to dig deeper in the topic of ``ImageNet`` training,
feel free to read `the next tutorial on ``ImageNet`` <>`__.

Or, if you would like to know how to train a powerful model on your own image data,
please go ahead and read `xxx <>`__.

.. |imagenet| image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/classification/imagenet_mosaic.jpg
.. |image0| image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/classification/mt_baker.jpg

"""
