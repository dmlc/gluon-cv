"""MS COCO Instance Segmentation Evaluate Metrics."""
from __future__ import absolute_import

import io
import sys
import warnings
from os import path as osp

import mxnet as mx
import numpy as np

try:
    from mxnet.metric import EvalMetric
except ImportError:
    from mxnet.gluon.metric import EvalMetric


class COCOInstanceMetric(EvalMetric):
    """Instance segmentation metric for COCO bbox and segm task.
    Will return box summary, box metric, seg summary and seg metric.

    Parameters
    ----------
    dataset : instance of gluoncv.data.COCOInstance
        The validation dataset.
    save_prefix : str
        Prefix for the saved JSON results.
    use_time : bool
        Append unique datetime string to created JSON file name if ``True``.
    cleanup : bool
        Remove created JSON file if ``True``. When cleanup is True, we do not output json to save
        computation.
    use_ext : bool
        Whether to use external module built by NVIDIA, which is significantly faster.
        Make sure NVIDIA MSCOCO API is installed!
    starting_id : int, default is 0
        If using distributed evaluation, it need to be set to the start of each shard.
    score_thresh : float
        Detection results with confident scores smaller than ``score_thresh`` will
        be discarded before saving to results.

    """

    def __init__(self, dataset, save_prefix, use_time=True, cleanup=False,
                 use_ext=False, starting_id=0, score_thresh=1e-3):
        self._starting_id = starting_id
        self._use_ext = use_ext
        super(COCOInstanceMetric, self).__init__('COCOInstance')
        self.dataset = dataset
        self._dump_to_file = not cleanup
        if use_time:
            import datetime
            t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        else:
            t = ''
        self._img_ids = sorted(dataset.coco.getImgIds())
        self._current_id = starting_id
        self._results = []
        self._score_thresh = score_thresh
        from ...data.mscoco.utils import try_import_pycocotools
        try_import_pycocotools()
        import pycocotools.mask as cocomask
        self._cocomask = cocomask
        self._filename = osp.abspath(osp.expanduser(save_prefix) + t + '.json')
        if use_ext:
            from pycocotools import coco
            if not hasattr(coco, 'ext'):
                raise AttributeError('external module is not support by the COCO API installed. '
                                     'Consider install NVIDIA MSCOCO API here '
                                     'https://github.com/NVIDIA/cocoapi.git')
            self.dataset.coco.createIndex(use_ext=use_ext)
            self._bbox_result = []
            self._segm_result = []
            self._bbox_filename = osp.abspath(osp.expanduser(save_prefix) + t + '_bbox.json')
            self._segm_filename = osp.abspath(osp.expanduser(save_prefix) + t + '_segm.json')
            if not self._dump_to_file:
                warnings.warn('When using external module, eval json is always dumped.')

    def get_result_buffer(self):
        if self._use_ext:
            return self._bbox_result, self._segm_result
        return self._results

    def reset(self):
        self._current_id = self._starting_id
        self._results = []
        if self._use_ext:
            self._bbox_result = []
            self._segm_result = []

    def _dump_json(self):
        """Write coco json file"""
        if not self._current_id == len(self._img_ids):
            warnings.warn(
                'Recorded {} out of {} validation images, incomplete results'.format(
                    self._current_id, len(self._img_ids)))
        import json
        try:
            if self._use_ext:
                with open(self._bbox_filename, 'w') as f:
                    json.dump(self._bbox_result, f)
                with open(self._segm_filename, 'w') as f:
                    json.dump(self._segm_result, f)
            else:
                with open(self._filename, 'w') as f:
                    json.dump(self._results, f)
        except IOError as e:
            raise RuntimeError("Unable to dump json file, ignored. What(): {}".format(str(e)))

    def _get_ap(self, coco_eval):
        """Return the default AP from coco_eval."""

        # Metric printing adapted from detectron/json_dataset_evaluator.
        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        return ap_default

    # pylint: disable=unexpected-keyword-arg
    def _update(self, annType='bbox'):
        """Use coco to get real scores. """
        if self._use_ext:
            self.dataset.coco.createIndex(use_ext=True)
            if annType == 'bbox':
                pred = self.dataset.coco.loadRes(self._bbox_filename, use_ext=True)
            else:
                pred = self.dataset.coco.loadRes(self._segm_filename, use_ext=True)
        else:
            if self._results is not None and not self._dump_to_file:
                pred = self.dataset.coco.loadRes(self._results)
            else:
                pred = self.dataset.coco.loadRes(self._filename)
        gt = self.dataset.coco
        # lazy import pycocotools
        from ...data.mscoco.utils import try_import_pycocotools
        try_import_pycocotools()
        from pycocotools.cocoeval import COCOeval
        # only NVIDIA MSCOCO API support use_ext
        coco_eval = COCOeval(gt, pred, annType, use_ext=self._use_ext) \
            if self._use_ext else COCOeval(gt, pred, annType)
        coco_eval.evaluate()
        coco_eval.accumulate()
        names, values = [], []
        names.append('~~~~ Summary {} metrics ~~~~\n'.format(annType))
        # catch coco print string, don't want directly print here
        _stdout = sys.stdout
        sys.stdout = io.StringIO()
        coco_eval.summarize()
        coco_summary = sys.stdout.getvalue()
        sys.stdout = _stdout
        values.append(str(coco_summary).strip())
        names.append('~~~~ Mean AP for {} ~~~~\n'.format(annType))
        values.append('{:.1f}'.format(100 * self._get_ap(coco_eval)))
        return names, values

    def get(self):
        """Get evaluation metrics. """
        if self._dump_to_file:
            self._dump_json()
        bbox_names, bbox_values = self._update('bbox')
        mask_names, mask_values = self._update('segm')
        names = bbox_names + mask_names
        values = bbox_values + mask_values
        return names, values

    def _encode_mask(self, mask):
        """Convert mask to coco rle"""
        rle = self._cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
        rle['counts'] = rle['counts'].decode('ascii')
        return rle

    # pylint: disable=arguments-differ, unused-argument
    def update(self, pred_bboxes, pred_labels, pred_scores, pred_masks, *args, **kwargs):
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
        pred_masks: mxnet.NDArray or numpy.ndarray
            Prediction masks with *original* shape `H, W`.

        """

        def as_numpy(a):
            """Convert a (list of) mx.NDArray into numpy.ndarray"""
            if isinstance(a, mx.nd.NDArray):
                a = a.asnumpy()
            return a

        # mask must be the same as image shape, so no batch dimension is supported
        pred_bbox, pred_label, pred_score, pred_mask = [
            as_numpy(x) for x in [pred_bboxes, pred_labels, pred_scores, pred_masks]]
        # filter out padded detection & low confidence detections
        valid_pred = np.where((pred_label >= 0) & (pred_score >= self._score_thresh))[0]
        pred_bbox = pred_bbox[valid_pred].astype('float32')
        pred_label = pred_label.flat[valid_pred].astype('int32')
        pred_score = pred_score.flat[valid_pred].astype('float32')
        pred_mask = pred_mask[valid_pred].astype('uint8')
        imgid = self._img_ids[int(self._current_id)]
        self._current_id += 1
        # for each bbox detection in each image
        for bbox, label, score, mask in zip(pred_bbox, pred_label, pred_score, pred_mask):
            if label not in self.dataset.contiguous_id_to_json:
                # ignore non-exist class
                continue
            if score < self._score_thresh:
                continue
            category_id = self.dataset.contiguous_id_to_json[label]
            # convert [xmin, ymin, xmax, ymax]  to [xmin, ymin, w, h]
            bbox[2:4] -= bbox[:2]
            # coco format full image mask to rle
            rle = self._encode_mask(mask)
            if self._use_ext:
                self._bbox_result.append({'image_id': int(imgid),
                                          'category_id': category_id,
                                          'bbox': list(map(lambda x: float(round(x, 2)), bbox[:4])),
                                          'score': float(round(score, 3))})
                self._segm_result.append({'image_id': int(imgid),
                                          'category_id': category_id,
                                          'segmentation': rle,
                                          'score': float(round(score, 3))})

            else:
                self._results.append({'image_id': int(imgid),
                                      'category_id': category_id,
                                      'bbox': list(map(lambda x: float(round(x, 2)), bbox[:4])),
                                      'score': float(round(score, 3)),
                                      'segmentation': rle})
