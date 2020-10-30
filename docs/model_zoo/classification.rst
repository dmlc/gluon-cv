.. _gluoncv-model-zoo-classification:

Classification
====================

.. role:: framework
   :class: framework
.. role:: select
   :class: selected framework

.. container:: Frameworks

  .. container:: framework-group

     :framework:`MXNet`
     :framework:`Pytorch`

.. rst-class:: MXNet

MXNet
*************

Visualization of Inference Throughputs vs. Validation Accuracy of ImageNet pre-trained models is illustrated in the following graph. Throughputs are measured with single V100 GPU and batch size 64.

.. image:: /_static/plot_help.png
  :width: 100%

.. raw:: html
   :file: ../_static/classification_throughputs.html

How To Use Pretrained Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The following example requires ``GluonCV>=0.4`` and ``MXNet>=1.4.0``. Please follow `our installation guide <../index.html#installation>`__ to install or upgrade GluonCV and MXNet if necessary.

- Prepare an image by yourself or use `our sample image <../_static/classification-demo.png>`__. You can save the image into filename ``classification-demo.png`` in your working directory or change the filename in the source codes if you use an another name.

- Use a pre-trained model. A model is specified by its name.

Let's try it out!

.. code-block:: python

    import mxnet as mx
    import gluoncv

    # you can change it to your image filename
    filename = 'classification-demo.png'
    # you may modify it to switch to another model. The name is case-insensitive
    model_name = 'ResNet50_v1d'
    # download and load the pre-trained model
    net = gluoncv.model_zoo.get_model(model_name, pretrained=True)
    # load image
    img = mx.image.imread(filename)
    # apply default data preprocessing
    transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img)
    # run forward pass to obtain the predicted score for each class
    pred = net(transformed_img)
    # map predicted values to probability by softmax
    prob = mx.nd.softmax(pred)[0].asnumpy()
    # find the 5 class indices with the highest score
    ind = mx.nd.topk(pred, k=5)[0].astype('int').asnumpy().tolist()
    # print the class name and predicted probability
    print('The input picture is classified to be')
    for i in range(5):
        print('- [%s], with probability %.3f.'%(net.classes[ind[i]], prob[ind[i]]))

The output from `our sample image <../_static/classification-demo.png>`__ is expected to be

.. code-block:: txt

    The input picture is classified to be
    - [Welsh springer spaniel], with probability 0.899.
    - [Irish setter], with probability 0.005.
    - [Brittany spaniel], with probability 0.003.
    - [cocker spaniel], with probability 0.002.
    - [Blenheim spaniel], with probability 0.002.


Remember, you can try different models by replacing the value of ``model_name``.
Read further for model names and their performances in the tables.

.. role:: greytag

ImageNet
~~~~~~~~

.. hint::

    Training commands work with this script:

    :download:`Download train_imagenet.py<../../scripts/classification/imagenet/train_imagenet.py>`

    A model can have differently trained parameters with different hashtags.
    Parameters with :greytag:`a grey name` can be downloaded by passing the corresponding hashtag.

    - Download default pretrained weights: ``net = get_model('ResNet50_v1d', pretrained=True)``

    - Download weights given a hashtag: ``net = get_model('ResNet50_v1d', pretrained='117a384e')``

    ``ResNet50_v1_int8`` and ``MobileNet1.0_int8`` are quantized model calibrated on ImageNet dataset.

.. role:: tag

ResNet
------

.. hint::

    - ``ResNet50_v1_int8`` is a quantized model for ``ResNet50_v1``.

    - ``ResNet_v1b`` modifies ``ResNet_v1`` by setting stride at the 3x3 layer for a bottleneck block.

    - ``ResNet_v1c`` modifies ``ResNet_v1b`` by replacing the 7x7 conv layer with three 3x3 conv layers.

    - ``ResNet_v1d`` modifies ``ResNet_v1c`` by adding an avgpool layer 2x2 with stride 2 downsample feature map on the residual path to preserve more information.

.. csv-table::
   :file: ./csv_tables/Classifications/ResNet.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 10 10 15 15 15

ResNext
-------

.. csv-table::
   :file: ./csv_tables/Classifications/ResNext.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 10 10 15 15 15

ResNeSt
-------

.. csv-table::
   :file: ./csv_tables/Classifications/ResNeSt.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 10 10 15 15 15

MobileNet
---------

.. hint::

    - ``MobileNet1.0_int8`` is a quantized model for ``MobileNet1.0``.

.. csv-table::
   :file: ./csv_tables/Classifications/MobileNet.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 10 10 15 15 15

VGG
---

.. csv-table::
   :file: ./csv_tables/Classifications/VGG.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 10 10 15 15 15

SqueezeNet
----------

.. csv-table::
   :file: ./csv_tables/Classifications/SqueezeNet.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 10 10 15 15 15

DenseNet
--------

.. csv-table::
   :file: ./csv_tables/Classifications/DenseNet.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 10 10 15 15 15

Pruned ResNet
-------------

.. csv-table::
   :file: ./csv_tables/Classifications/Pruned_ResNet.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 10 10 15 30

Others
------

.. hint::

    ``InceptionV3`` is evaluated with input size of 299x299.

.. csv-table::
   :file: ./csv_tables/Classifications/Others.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 10 10 15 15 15

CIFAR10
~~~~~~~

The following table lists pre-trained models trained on CIFAR10.

.. hint::

    Our pre-trained models reproduce results from "Mix-Up" [13]_ .
    Please check the reference paper for further information.

    Training commands in the table work with the following scripts:

    - For vanilla training: :download:`Download train_cifar10.py<../../scripts/classification/cifar/train_cifar10.py>`
    - For mix-up training: :download:`Download train_mixup_cifar10.py<../../scripts/classification/cifar/train_mixup_cifar10.py>`

.. csv-table::
   :file: ./csv_tables/Classifications/CIFAR10.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 25 25 25

.. [1] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. \
       "Deep residual learning for image recognition." \
       In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.
.. [2] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. \
       "Identity mappings in deep residual networks." \
       In European Conference on Computer Vision, pp. 630-645. Springer, Cham, 2016.
.. [3] Redmon, Joseph, and Ali Farhadi. \
       "Yolov3: An incremental improvement." \
       arXiv preprint arXiv:1804.02767 (2018).
.. [4] Howard, Andrew G., Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, \
       Tobias Weyand, Marco Andreetto, and Hartwig Adam. \
       "Mobilenets: Efficient convolutional neural networks for mobile vision applications." \
       arXiv preprint arXiv:1704.04861 (2017).
.. [5] Sandler, Mark, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. \
       "Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation." \
       arXiv preprint arXiv:1801.04381 (2018).
.. [6] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. \
       "Imagenet classification with deep convolutional neural networks." \
       In Advances in neural information processing systems, pp. 1097-1105. 2012.
.. [7] Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. \
       "Densely Connected Convolutional Networks." In CVPR, vol. 1, no. 2, p. 3. 2017.
.. [8] Szegedy, Christian, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. \
       "Rethinking the inception architecture for computer vision." \
       In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2818-2826. 2016.
.. [9] Karen Simonyan, Andrew Zisserman. \
       "Very Deep Convolutional Networks for Large-Scale Image Recognition." \
       arXiv technical report arXiv:1409.1556 (2014).
.. [10] Iandola, Forrest N., Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, and Kurt Keutzer. \
       "Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 0.5 mb model size." \
       arXiv preprint arXiv:1602.07360 (2016).
.. [11] Zagoruyko, Sergey, and Nikos Komodakis. \
       "Wide residual networks." \
       arXiv preprint arXiv:1605.07146 (2016).
.. [12] Xie, Saining, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. \
       "Aggregated residual transformations for deep neural networks." \
       In Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on, pp. 5987-5995. IEEE, 2017.
.. [13] Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. \
       "mixup: Beyond empirical risk minimization." \
       arXiv preprint arXiv:1710.09412 (2017).
.. [14] Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." arXiv preprint arXiv:1709.01507 7 (2017).
.. [15] Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al. \
       "Searching for mobilenetv3." \
       arXiv preprint arXiv:1905.02244 (2019).
.. [16] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich \
        "Going Deeper with Convolutions" \
        arXiv preprint arXiv:1409.4842 (2014).
.. [17] Hang Zhang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Muller, R. Manmatha, Mu Li and Alex Smola \
        "ResNeSt: Split-Attention Network" \
        arXiv preprint (2020).


.. rst-class:: Pytorch

Pytorch
*************
