import argparse
import time
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet import nd, gluon
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

def parse_args():
    parser = argparse.ArgumentParser(description='Test Faster RCNN.')
    parser.add_argument('--network', type=str, default='resnet101',
                        help="Base network name")
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    args = parser.parse_args()


def get_dataset(dataset):
    if dataset.lower() == 'voc':
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset


def get_dataloader(val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    val_loader = gdata.DetectionDataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, last_batch='keep', num_workers=num_workers)
    return val_loader


if __name__ == '__main__':
    args = parse_args()
    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    # validation data
    val_dataset = get_dataset(args.dataset)
    val_data = get_dataloader(
        val_dataset, args.data_shape, args.batch_size, args.num_workers)
    classes = val_dataset.classes
    # network
    net = gcv.model_zoo.FasterRCNN(20, 'resnet101')
