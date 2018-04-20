
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet.test_utils import assert_almost_equal
from mxnet.gluon.data.vision import transforms

import gluonvision
from gluonvision.utils.metrics.voc_segmentation import *
from gluonvision.data import VOCSegmentation

ctx = mx.gpu(0)
try:
    x = mx.ones((1)).as_in_context(ctx)
except Exception:
    ctx = mx.cpu(0)

def test_segmentation_utils():
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    # get the dataset
    # TODO FIXME: change it to ADE20K dataset and pretrained model
    dataset = VOCSegmentation(split='val')
    # load pretrained net
    net = gluonvision.model_zoo.get_model('fcn_resnet50_voc', pretrained=True, ctx=ctx)
    # count for pixAcc and mIoU
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    np_inter, np_union, np_correct, np_label = 0, 0, 0, 0
    tbar = tqdm(range(100))
    for i in tbar:
        img, mask = dataset[i]
        # prepare data and make prediction
        img = transform_fn(img)
        img = img.expand_dims(0).as_in_context(ctx)
        mask = mask.expand_dims(0)
        pred = net.evaluate(img).as_in_context(mx.cpu(0))
        # gv prediction
        correct1, labeled1 = batch_pix_accuracy(pred, mask, True)
        inter1, union1 = batch_intersection_union(pred, mask, dataset.num_class, True)
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
        inter2, union2 = intersectionAndUnion(pred, mask, dataset.num_class-1)
        np_correct += correct2
        np_label += labeled2
        np_inter += inter2
        np_union += union2
        np_pixAcc = 1.0 * np_correct / (np.spacing(1) + np_label)
        np_IoU = 1.0 * np_inter / (np.spacing(1) + np_union)
        np_mIoU = np_IoU.mean()
        # logging
        tbar.set_description('pixAcc: %.3f, np_pixAcc: %.3f, mIoU: %.3f,  np_mIoU: %.3f'%\
            (pixAcc, np_pixAcc, mIoU, np_mIoU))

    assert(total_inter == np_inter).all()
    assert(total_union == np_union).all()
    assert(total_correct == np_correct).all()
    assert(total_label == np_label).all()


if __name__ == '__main__':
    import nose
    nose.runmodule()
