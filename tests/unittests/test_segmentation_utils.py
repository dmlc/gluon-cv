
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet.test_utils import assert_almost_equal

from gluonvision.utils.metrics.voc_segmentation import *
from gluonvision.data import ADE20KSegmentation

def test_segmentation_utils():
    dataset = ADE20KSegmentation(split='val')
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    np_inter, np_union, np_correct, np_label = 0, 0, 0, 0
    tbar = tqdm(range(10))
    for i in tbar:
        _, mask = dataset[i]
        mask = mask.expand_dims(0)
        # mask shape 1xHxW
        pred = mask.one_hot(depth=151).swapaxes(dim1=1, dim2=3).swapaxes(dim1=2, dim2=3)
        # mask shape 1xHxWxN
        rand = mx.random.uniform(shape=pred.shape) > 0.0001
        pred = mx.nd.where(rand, pred, 2*mx.nd.ones_like(pred))
        # gv prediction
        correct1, labeled1 = batch_pix_accuracy(pred, mask, True)
        inter1, union1 = batch_intersection_union(pred, mask, 151, True)
        total_correct += correct1
        total_label += labeled1
        total_inter += inter1
        total_union += union1

        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()

        # np predicition
        pred = mx.nd.squeeze(mx.nd.argmax(pred, 1)).asnumpy()
        mask = mask.squeeze().asnumpy()
        _, correct2, labeled2 = pixelAccuracy(pred, mask)
        inter2, union2 = intersectionAndUnion(pred, mask, 150)
        np_correct += correct2
        np_label += labeled2
        np_inter += inter2
        np_union += union2
    
        np_pixAcc = 1.0 * np_correct / (np.spacing(1) + np_label)
        np_IoU = 1.0 * np_inter / (np.spacing(1) + np_union)
        np_mIoU = np_IoU.mean()

        tbar.set_description('pixAcc: %.3f, np_pixAcc: %.3f, mIoU: %.3f,  np_mIoU: %.3f'%\
            (pixAcc, np_pixAcc, mIoU, np_mIoU))

    assert(total_inter == np_inter).all()
    assert(total_union == np_union).all()
    assert(total_correct == np_correct).all()
    assert(total_label == np_label).all()


if __name__ == '__main__':
    import nose
    nose.runmodule()
