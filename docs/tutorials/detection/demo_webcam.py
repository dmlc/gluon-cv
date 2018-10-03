"""8. Run an object detection model on your webcam
==================================================

This article will shows how to play with pre-trained object detection models by running
them directly on your webcam video stream.

.. note::

    - This tutorial has only been tested in a MacOS environment
    - Python packages required: cv2, matplotlib
    - You need a webcam :)
    - Python compatible with matplotlib rendering, installed as a framework in MacOS see guide `here <https://matplotlib.org/faq/osx_framework.html>`__


Loading the model and webcam
----------------------------
Finished preparation? Let's get started!
First, import the necessary libraries into python.

.. code-block:: python

    import time

    import cv2
    import gluoncv as gcv
    import matplotlib.pyplot as plt
    import mxnet as mx
    import numpy as np


In this tutorial we use ``ssd_512_mobilenet1.0_voc``, a snappy network with good accuracy that should be
well above 1 frame per second on most laptops. Feel free to try a different model from
the :doc:`../../model_zoo/index` !

.. code-block:: python

    # Load the model
    net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)


We create the webcam handler in opencv to be able to acquire the frames:

.. code-block:: python

    # Load the webcam handler
    cap = cv2.VideoCapture(0)
    time.sleep(1) ### letting the camera autofocus


Detection loop
--------------

The detection loop consists of four phases:

* loading the webcam frame

* pre-processing the image:

    * resizing the image

    * converting the image from BGR to RGB

    * normalizing the data and converting it to channel first

* running the image through the network

* updating the output with the resulting predictions


.. code-block:: python

    NUM_FRAMES = 200 # you can change this
    for i in range(NUM_FRAMES):
        # Load frame from the camera
        ret, frame = cap.read()

        # Image pre-processing
        frame = cv2.resize(frame, (700, 512))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_nd = mx.nd.array(rgb, dtype='float32').transpose((2, 0, 1))/255.
        rgb_nd = mx.nd.image.normalize(rgb_nd, mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)).expand_dims(axis=0)

        # Run frame through network
        class_IDs, scores, bounding_boxes = net(rgb_nd)

        # Display the result
        plt.cla()
        axes = gcv.utils.viz.plot_bbox(rgb, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes, ax=axes)
        plt.draw()
        plt.pause(0.001)


We release the webcam before exiting the script

.. code-block:: python

    cap.release()

Results
---------
Copy the content of the above code blocks into a file called `demo_webcam_run.py`.
Run the script using `pythonw` on MacOS:

.. code-block:: bash

    pythonw demo_webcam_run.py


.. note::

    On MacOS, to enable matplotlib rendering you need python installed as a framework,
    see guide `here <https://matplotlib.org/faq/osx_framework.html>`__


If all goes well you should be able to detect objects from the available
classes of the VOC dataset. That includes persons, chairs and TV Screens!

.. image:: https://media.giphy.com/media/9JvoKeUeCt4bdRf3Cv/giphy.gif


"""
