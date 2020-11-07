"""3. Getting Started with Pre-trained Models on ImageNet
===========================================================

`ImageNet <http://www.image-net.org/>`__ is a
large labeled dataset of real-world images. It is one of the most
widely used dataset in latest computer vision research.

|imagenet|

In this tutorial, we will show how a pre-trained neural network
classifies real world images.

For your convenience, we provide a script that loads a pre-trained ``ResNet50_v2`` model,
and classifies an input image.
For a list of all models we have, please visit `Gluon Model Zoo <../../model_zoo/index.html>`__.

Demo
------------------

A model trained on ImageNet can classify images into 1000 classes, this makes it
much more powerful than the one we showed in the `CIFAR10 demo <demo_cifar10.html>`__.

:download:`Download demo_imagenet.py<../../../scripts/classification/imagenet/demo_imagenet.py>`

With this script, you can load a pre-trained model and classify any image you have.

Let's test with the photo of Mt. Baker again.

|image0|

::

    python demo_imagenet.py --model ResNet50_v2 --input-pic mt_baker.jpg

And the model predicts that

::

    The input picture is classified to be
    	[volcano], with probability 0.558.
    	[alp], with probability 0.398.
    	[valley], with probability 0.018.
    	[lakeside], with probability 0.006.
    	[mountain_tent], with probability 0.006.

This time it does a good job. Note that we have listed the top five
most probable classes, because with 1000 classes the model may not always rank the
correct answer highest. Besides top-1 accuracy, we often also
consider top-5 accuracy as a measurement of how well a model can predict.

Next Step
---------

If you would like to dive deeper into ``ImageNet`` training,
feel free to read the next tutorial on `ImageNet Training <dive_deep_imagenet.html>`__.

Or, if you would like to know how to train a powerful model tailored to your own data,
please go ahead and read the tutorial on `Transfer learning <transfer_learning_minc.html>`__.

.. |imagenet| image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/datasets/imagenet_mosaic.jpg
.. |image0| image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/classification/mt_baker.jpg

"""
