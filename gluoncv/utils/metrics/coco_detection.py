"""MS COCO Detection Evaluate Metrics."""
from __future__ import absolute_import

from os import path as osp
import numpy as np
import mxnet as mx


class COCOBBoxMetric(mx.metirc.EvalMetric):
    def __init__(self, dataset, save_prefix, use_time=True, cleanup=False):
        self.dataset = dataset
        self._img_ids = sorted(dataset.coco.getImgIds())
        self._current_id = 0
        if use_time:
            import datatime
            t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        else:
            t = ''
        self._filename = osp.abs_path(osp.expanduser(prefix) + t + '.json')
        try:
            f = open(self._filename, 'w')
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()
        self._cleanup = cleanup
        self._results = []

    def __del__(self):
        if self._cleanup:
            try:
                os.remove(self._filename)
            except IOError as err:
                import warnings
                warnings.warn(str(err))

    def reset(self):
        self._current_id = 0
        self._results = []

    def _update(self):
        """Use coco to get real scores. """
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
        from pycocotools.coco import COCOeval
        coco_eval = C

        aps = []
        recall, precs = self._recall_prec()
        for l, rec, prec in zip(range(len(precs)), recall, precs):
            ap = self._average_precision(rec, prec)
            aps.append(ap)
            if self.num is not None and l < (self.num - 1):
                self.sum_metric[l] = ap
                self.num_inst[l] = 1
        if self.num is None:
            self.num_inst = 1
            self.sum_metric = np.nanmean(aps)
        else:
            self.num_inst[-1] = 1
            self.sum_metric[-1] = np.nanmean(aps)


    def update(self, pred_bboxes, pred_labels, pred_scores):
        """Update internal buffer with latest predictions.
        Note that the statistics are not available until you call self.get() to return
        the metrics.

        Parameters
        ----------
        pred_bboxes : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes with shape `B, N, 4`.
            Where B is the size of mini-batch, N is the number of bboxes.
        pred_labels : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes labels with shape `B, N`.
        pred_scores : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes scores with shape `B, N`.

        """
        for pred_bbox, pred_label, pred_score in zip(
            *[x.asnumpy() if isinstance(x, mx.nd.NDArray) else x \
                for x in [pred_bboxes, pred_labels, pred_scores]]):
            valid_pred = np.where(pred_label.flat >= 0)[0]
            pred_bbox = pred_bbox[valid_pred, :].astype(np.float)
            pred_label = pred_label.flat[valid_pred].astype(int)
            pred_score = pred_score.flat[valid_pred].astype(np.float)

            imgid = self._img_ids[self._current_id]
            self._current_id += 1
            # for each bbox detection in each image
            for bbox, label, score in zip(pred_bbox, pred_label, pred_score):
                if label not in self.dataset.contiguous_id_to_json:
                    # ignore non-exist class
                    continue
                category_id = self.dataset.contiguous_id_to_json[label]
                self._results.append({'image_id': imgid,
                                      'category_id': category_id,
                                      'bbox': bbox[:4].tolist(),
                                      'score': score})
