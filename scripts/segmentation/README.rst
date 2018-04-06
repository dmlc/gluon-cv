Semantic Segmentation
=====================

Test Pre-trained Model
~~~~~~~~~~~~~~~~~~~~~~



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
