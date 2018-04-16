Benchmarks and Training
_______________________
.. code-block:: python

    # load pre-trained model
    model = FCN(nclass=22, backbone='resnet101')
    model.load_params('fcn101.params')
    model.collect_params().reset_ctx(ctx)

    # read image and normalize the data
    transform = Compose([
        ToTensor(ctx=ctx),
        Normalize(mean=[.485, .456, .406], std=[.229, .224, .225], ctx=ctx)])

    def load_image(path, transform, ctx):
        image = Image.open(path).convert('RGB')
        image = transform(image)
        image = image.expand_dims(0).as_in_context(ctx)
        return image

    image = load_image('example.jpg', transform, ctx)

    # make prediction using single scale
    output = model(image)
    predict = F.squeeze(F.argmax(output, 1)).asnumpy()

    # add color pallete for visualization
    mask = get_mask(predict, 'pascal_voc')
    mask.save('output.png')

Please see the demo.py for more evaluation options.


Benchmarks and Training
_______________________

- Checkout the training scripts for reproducing the experiments, and see the detail running 
instructions in the `README <https://github.com/dmlc/gluon-vision/tree/master/scripts/segmentation>`_ .

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

References
----------

.. [Long15] Long, Jonathan, Evan Shelhamer, and Trevor Darrell. \
    "Fully convolutional networks for semantic segmentation." \
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
