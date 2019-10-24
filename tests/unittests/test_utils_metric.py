from __future__ import print_function

import numpy as np
import gluoncv as gcv
import mxnet as mx
from mxnet import autograd, gluon
from gluoncv.utils import download, viz

def test_voc07_metric_difficult():
    url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/train.rec'
    idx_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/train.idx'
    download(url, path='pikachu_train.rec', overwrite=False)
    download(idx_url, path='pikachu_train.idx', overwrite=False)
    classes = ['pikachu']

    dataset = gcv.data.RecordFileDetection('pikachu_train.rec')
    net = gcv.model_zoo.get_model('yolo3_darknet53_custom', classes=classes,
                                  pretrained_base=False, transfer='voc')

    def get_dataloader(val_dataset, data_shape, batch_size, num_workers):
        from gluoncv.data.batchify import Tuple, Stack, Pad
        from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform

        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = gluon.data.DataLoader(
            val_dataset.transform(YOLO3DefaultValTransform(data_shape, data_shape)),
            batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
        return val_loader

    val_data = get_dataloader(dataset, 416, 16, 0)

    try:
        a = mx.nd.zeros((1,), ctx=mx.gpu(0))
        ctx = [mx.gpu(0)]
    except:
        ctx = [mx.cpu()]

    from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

    val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)

    def validate(net, val_data, ctx, eval_metric):
        """Test on validation dataset."""
        eval_metric.reset()
        # set nms threshold and topk constraint
        net.set_nms(nms_thresh=0.45, nms_topk=400)
        mx.nd.waitall()
        net.hybridize()
        for batch in val_data:
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(data, label):

                # get prediction results
                ids, scores, bboxes = net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            # update metric
            eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
            break
        return eval_metric.get()

    net.collect_params().reset_ctx(ctx)
    validate(net, val_data, ctx, val_metric)


if __name__ == '__main__':
    import nose
    nose.runmodule()
