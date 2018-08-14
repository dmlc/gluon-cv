"""Mask RCNN Demo script."""
import os
import argparse
import mxnet as mx
import gluoncv as gcv
from gluoncv.data.transforms import presets
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Test with Mask RCNN networks.')
    parser.add_argument('--network', type=str, default='mask_rcnn_resnet50_v1b_coco',
                        help="Mask RCNN full network name")
    parser.add_argument('--images', type=str, default='',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters. You can specify parameter file name.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    # grab some image if not specified
    if not args.images.strip():
        gcv.utils.download('https://github.com/dmlc/web-data/blob/master/' +
                           'gluoncv/detection/biking.jpg?raw=true', 'biking.jpg')
        image_list = ['biking.jpg']
    else:
        image_list = [x.strip() for x in args.images.split(',') if x.strip()]

    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(args.network, pretrained=True)
    else:
        net = gcv.model_zoo.get_model(args.network, pretrained=False)
        net.load_parameters(args.pretrained)
    net.set_nms(0.3, 200)

    for image in image_list:
        x, img = presets.rcnn.load_test(image, short=net.short, max_size=net.max_size)
        ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

        masks = gcv.utils.viz.expand_mask(masks, bboxes, (img.shape[1], img.shape[0]), scores)
        img = gcv.utils.viz.plot_mask(img, masks)

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(1, 1, 1)
        ax = gcv.utils.viz.plot_bbox(img, bboxes, scores, ids,
                                     class_names=net.classes, ax=ax)
        plt.show()
