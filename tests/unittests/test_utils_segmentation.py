from tqdm import tqdm
import numpy as np
import mxnet as mx
from mxnet.test_utils import assert_almost_equal
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.utils.metrics.segmentation import *
from gluoncv.data import ADE20KSegmentation

from common import try_gpu, with_cpu

@with_cpu(0)
def test_segmentation_utils():
    ctx = mx.context.current_context()
    import os
    if not os.path.isdir(os.path.expanduser('~/.mxnet/datasets/voc')):
        return

    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    # get the dataset
    # TODO FIXME: change it to ADE20K dataset and pretrained model
    dataset = ADE20KSegmentation(split='val')
    # load pretrained net
    net = gluoncv.model_zoo.get_model('fcn_resnet50_ade', pretrained=True, ctx=ctx)
    # count for pixAcc and mIoU
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    np_inter, np_union, np_correct, np_label = 0, 0, 0, 0
    tbar = tqdm(range(10))
    for i in tbar:
        img, mask = dataset[i]
        # prepare data and make prediction
        img = transform_fn(img)
        img = img.expand_dims(0).as_in_context(ctx)
        mask = mask.expand_dims(0)
        pred = net.evaluate(img).as_in_context(mx.cpu(0))
        # gcv prediction
        correct1, labeled1 = batch_pix_accuracy(pred, mask)
        inter1, union1 = batch_intersection_union(pred, mask, dataset.num_class)
        total_correct += correct1
        total_label += labeled1
        total_inter += inter1
        total_union += union1
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()

        # np predicition
        pred2 = np.argmax(pred.asnumpy().astype('int64'), 1) + 1
        mask2 = mask.squeeze().asnumpy().astype('int64') + 1
        _, correct2, labeled2 = pixelAccuracy(pred2, mask2)
        inter2, union2 = intersectionAndUnion(pred2, mask2, dataset.num_class)
        np_correct += correct2
        np_label += labeled2
        np_inter += inter2
        np_union += union2
        np_pixAcc = 1.0 * np_correct / (np.spacing(1) + np_label)
        np_IoU = 1.0 * np_inter / (np.spacing(1) + np_union)
        np_mIoU = np_IoU.mean()
        tbar.set_description('pixAcc: %.3f, np_pixAcc: %.3f, mIoU: %.3f, np_mIoU: %.3f'%\
            (pixAcc, np_pixAcc, mIoU, np_mIoU))

    np.testing.assert_allclose(total_inter, np_inter)
    np.testing.assert_allclose(total_union, np_union)
    np.testing.assert_allclose(total_correct, np_correct)
    np.testing.assert_allclose(total_label, np_label)

if __name__ == '__main__':
    import nose
    nose.runmodule()
