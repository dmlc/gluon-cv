Semantic Segmentation Tutorial
==============================

Tutorial and Examples
_____________________

This is a semantic segmentation tutorial using Gluon Vison, a step-by-step example.
The readers should have basic knowledge of deep learning and should be familiar with Gluon API.
New users may first go through Gluon tutorials
`Deep Learning - The Straight Dope<http://gluon.mxnet.io/>`_.



Benchmarks and Training
_______________________

Test Pre-trained Model
~~~~~~~~~~~~~~~~~~~~~~

- Table of pre-trained models and its performance (models :math:`^\ast` denotes pre-trained on COCO):

.. role:: raw-html(raw)
   :format: html

.. _Table:

    +------------------------+------------+-----------+-----------+-----------+-----------+----------------------------------------------------------------------------------------------+
    | Method                 | Backbone   | Dataset   | Note      | pixAcc    | mIoU      | Training Scripts                                                                             |
    +========================+============+===========+===========+===========+===========+==============================================================================================+
    | FCN                    | ResNet50   | PASCAL12  | stride 8  | N/A       | 70.9_     | :raw-html:`<a href="javascript:toggleblock('cmd_fcn_50')" class="toggleblock">cmd</a>`       |
    +------------------------+------------+-----------+-----------+-----------+-----------+----------------------------------------------------------------------------------------------+
    | FCN                    | ResNet101  | PASCAL12  | stride 8  | N/A       |           | :raw-html:`<a href="javascript:toggleblock('cmd_fcn_101')" class="toggleblock">cmd</a>`      |
    +------------------------+------------+-----------+-----------+-----------+-----------+----------------------------------------------------------------------------------------------+
    | PSPNet                 | ResNet50   | PASCAL12  | w/o aux   | N/A       |           | :raw-html:`<a href="javascript:toggleblock('cmd_psp_50')" class="toggleblock">cmd</a>`       |
    +------------------------+------------+-----------+-----------+-----------+-----------+----------------------------------------------------------------------------------------------+

    .. _70.9:  http://host.robots.ox.ac.uk:8080/anonymous/FR9APO.html

.. raw:: html

    <code xml:space="preserve" id="cmd_fcn_50" style="display: none; text-align: left; white-space: pre-wrap">
    # First training on augmented set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset pascal_aug --model fcn --backbone resnet50 --lr 0.001 --syncbn --checkname mycheckpoint
    # Finetuning on original set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset pascal_voc --model fcn --backbone resnet50 --lr 0.0001 --syncbn --checkname mycheckpoint --resume runs/pascal_aug/fcn/mycheckpoint/checkpoint.params
    </code>

    <code xml:space="preserve" id="cmd_fcn_101" style="display: none; text-align: left; white-space: pre-wrap">
    # First training on augmented set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset pascal_aug --model fcn --backbone resnet101 --lr 0.001 --syncbn --checkname mycheckpoint
    # Finetuning on original set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset pascal_voc --model fcn --backbone resnet101 --lr 0.0001 --syncbn --checkname mycheckpoint --resume runs/pascal_aug/fcn/mycheckpoint/checkpoint.params
    </code>

    <code xml:space="preserve" id="cmd_psp_50" style="display: none; text-align: left; white-space: pre-wrap">
    # First training on augmented set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset pascal_aug --model pspnet --backbone resnet50 --lr 0.001 --syncbn --checkname mycheckpoint
    # Finetuning on original set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset pascal_voc --model pspnet --backbone resnet50 --lr 0.0001 --syncbn --checkname mycheckpoint --resume runs/pascal_aug/fcn/mycheckpoint/checkpoint.params
    </code>

    <code xml:space="preserve" id="cmd_psp_101" style="display: none; text-align: left; white-space: pre-wrap">
    # First training on augmented set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset pascal_aug --model pspnet --backbone resnet101 --lr 0.001 --syncbn --checkname mycheckpoint
    # Finetuning on original set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset pascal_voc --model pspnet --backbone resnet101 --lr 0.0001 --syncbn --checkname mycheckpoint --resume runs/pascal_aug/fcn/mycheckpoint/checkpoint.params
    </code>

    <code xml:space="preserve" id="cmd_psp_101_coco" style="display: none; text-align: left; white-space: pre-wrap">
    # Pre-training on COCO dataset
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset mscoco --model pspnet --backbone resnet101 --lr 0.01 --syncbn --checkname mycheckpoint
    # Training on augmented set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset pascal_aug --model pspnet --backbone resnet101 --lr 0.001 --syncbn --checkname mycheckpoint
    # Finetuning on original set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset pascal_voc --model pspnet --backbone resnet101 --lr 0.0001 --syncbn --checkname mycheckpoint --resume runs/pascal_aug/fcn/mycheckpoint/checkpoint.params
    </code>


Train Your Own Model
~~~~~~~~~~~~~~~~~~~~

- Prepare PASCAL VOC Dataset and Augmented Dataset::

    cd examples/datasets/
    python setup_pascal_voc.py
    python setup_pascal_aug.py

- Training command example::

    # First training on augmented set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset pascal_aug --model fcn --backbone resnet50 --lr 0.001 --checkname mycheckpoint
    # Finetuning on original set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset pascal_voc --model fcn --backbone resnet50 --lr 0.0001 --checkname mycheckpoint --resume runs/pascal_aug/fcn/mycheckpoint/checkpoint.params

  For more training commands, please see the ``Commands`` in the pre-trained Table_.

- Detail training options::
    
    -h, --help            show this help message and exit
    --model MODEL         model name (default: fcn)
    --backbone BACKBONE   backbone name (default: resnet50)
    --dataset DATASET     dataset name (default: pascal)
    --nclass NCLASS       nclass for pre-trained model (default: None)
    --workers N           dataloader threads
    --data-folder         training dataset folder (default: $(HOME)/data/)
    --epochs N            number of epochs to train (default: 50)
    --start_epoch N       start epochs (default:0)
    --batch-size N        input batch size for training (default: 16)
    --test-batch-size N   input batch size for testing (default: 32)
    --lr LR               learning rate (default: 1e-3)
    --momentum M          momentum (default: 0.9)
    --weight-decay M      w-decay (default: 1e-4)
    --kvstore KVSTORE     kvstore to use for trainer/module.
    --no-cuda             disables CUDA training
    --ngpus NGPUS         number of GPUs (default: 4)
    --seed S              random seed (default: 1)
    --resume RESUME       put the path to resuming file if needed
    --checkname           set the checkpoint name
    --eval                evaluating mIoU
    --test                test a set of images and save the prediction
    --syncbn              using Synchronized Cross-GPU BatchNorm

Extending the Software
~~~~~~~~~~~~~~~~~~~~~~

- Write your own Dataloader ``mydataset.py`` to ``gluonvision/datasets/`` folder

- Write your own Model ``mymodel.py`` to ``gluonvision/models/`` folder

- Run the program:

.. code:: python

    python main.py --dataset mydataset --model mymodel --nclass 10 ...
