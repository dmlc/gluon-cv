"""MS COCO Key Points Evaluate Metrics."""
from __future__ import absolute_import

import sys
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import os
from os import path as osp
from collections import OrderedDict
import warnings
import numpy as np
import mxnet as mx
from ...data.mscoco.utils import try_import_pycocotools


class COCOKeyPointsMetric(mx.metric.EvalMetric):
    """Detection metric for COCO bbox task.

    Parameters
    ----------
    dataset : instance of gluoncv.data.COCODetection
        The validation dataset.
    save_prefix : str
        Prefix for the saved JSON results.
    use_time : bool
        Append unique datetime string to created JSON file name if ``True``.
    cleanup : bool
        Remove created JSON file if ``True``.
    score_thresh : float
        Detection results with confident scores smaller than ``score_thresh`` will
        be discarded before saving to results.
    data_shape : tuple of int, default is None
        If `data_shape` is provided as (height, width), we will rescale bounding boxes when
        saving the predictions.
        This is helpful when SSD/YOLO box predictions cannot be rescaled conveniently. Note that
        the data_shape must be fixed for all validation images.

    """
    def __init__(self, dataset, save_prefix, use_time=True, cleanup=False, score_thresh=0.05,
                 data_shape=None):
        super(COCOKeyPointsMetric, self).__init__('COCOMeanAP')
        self.dataset = dataset
        self._img_ids = sorted(dataset.coco.getImgIds())
        self._current_id = 0
        self._cleanup = cleanup
        self._results = []
        self._score_thresh = score_thresh
        if isinstance(data_shape, (tuple, list)):
            assert len(data_shape) == 2, "Data shape must be (height, width)"
        elif not data_shape:
            data_shape = None
        else:
            raise ValueError("data_shape must be None or tuple of int as (height, width)")
        self._data_shape = data_shape

        if use_time:
            import datetime
            t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        else:
            t = ''
        self._filename = osp.abspath(osp.expanduser(save_prefix) + t + '.json')
        try:
            f = open(self._filename, 'w')
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()

    def __del__(self):
        if self._cleanup:
            try:
                os.remove(self._filename)
            except IOError as err:
                warnings.warn(str(err))

    def reset(self):
        self._current_id = 0
        self._results = []

    def _update(self):
        """Use coco to get real scores. """
        if not self._current_id == len(self._img_ids):
            warnings.warn(
                'Recorded {} out of {} validation images, incompelete results'.format(
                    self._current_id, len(self._img_ids)))
        import json
        try:
            with open(self._filename, 'w') as f:
                json.dump(self._results, f)
        except IOError as e:
            raise RuntimeError("Unable to dump json file, ignored. What(): {}".format(str(e)))

        pred = self.dataset.coco.loadRes(self._filename)
        gt = self.dataset.coco
        # lazy import pycocotools
        try_import_pycocotools()
        from pycocotools.cocoeval import COCOeval
        coco_eval = COCOeval(gt, pred, 'keypoints')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self._coco_eval = coco_eval
        return coco_eval

    def get(self):
        """Get evaluation metrics. """
        # Metric printing adapted from detectron/json_dataset_evaluator.
        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        # call real update
        coco_eval = self._update()

        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))
        name_value = OrderedDict(info_str)
        return name_value, name_value['AP']

    # pylint: disable=arguments-differ, unused-argument
    def update(self, preds, maxvals, score, imgid, *args, **kwargs):
        batch_size = preds.shape[0]
        num_joints = preds.shape[1]
        for idx, kpt in enumerate(preds):
            kpt = []
            for i in range(num_joints):
                kpt += preds[idx][i].asnumpy().tolist()
                kpt.append(float(maxvals[idx][i].asscalar()))
            self._results.append({'image_id': int(imgid[idx].asscalar()),
                                  'category_id': 1,
                                  'keypoints': kpt,
                                  'score': int(score[idx].asscalar())})

