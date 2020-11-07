"""MS COCO Key Points Evaluate Metrics."""
from __future__ import absolute_import

import os
from os import path as osp
from collections import OrderedDict
import warnings
try:
    from mxnet.metric import EvalMetric
except ImportError:
    from mxnet.gluon.metric import EvalMetric

class COCOKeyPointsMetric(EvalMetric):
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
    in_vis_thresh : float
        Detection results with confident scores smaller than ``in_vis_thresh`` will
        be discarded before saving to results.
    data_shape : tuple of int, default is None
        If `data_shape` is provided as (height, width), we will rescale bounding boxes when
        saving the predictions.
        This is helpful when SSD/YOLO box predictions cannot be rescaled conveniently. Note that
        the data_shape must be fixed for all validation images.

    """
    def __init__(self, dataset, save_prefix, use_time=True, cleanup=False, in_vis_thresh=0.2,
                 data_shape=None):
        super(COCOKeyPointsMetric, self).__init__('COCOMeanAP')
        self.dataset = dataset
        self._img_ids = sorted(dataset.coco.getImgIds())
        self._recorded_ids = {}
        self._cleanup = cleanup
        self._results = []
        self._in_vis_thresh = in_vis_thresh
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
        self._recorded_ids = {}
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
        from ...data.mscoco.utils import try_import_pycocotools
        try_import_pycocotools()
        from pycocotools.cocoeval import COCOeval
        coco_eval = COCOeval(gt, pred, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self._coco_eval = coco_eval
        return coco_eval

    def get(self):
        """Get evaluation metrics. """
        # call real update
        coco_eval = self._update()

        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                       'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))
        name_value = OrderedDict(info_str)
        return name_value, name_value['AP']

    # pylint: disable=arguments-differ, unused-argument, missing-docstring
    def update(self, preds, maxvals, score, imgid, *args, **kwargs):
        # import pdb; pdb.set_trace()
        num_joints = preds.shape[1]
        in_vis_thresh = self._in_vis_thresh
        for idx, kpt in enumerate(preds):
            kpt = []
            kpt_score = 0
            count = 0
            for i in range(num_joints):
                kpt += preds[idx][i].asnumpy().tolist()
                mval = float(maxvals[idx][i].asscalar())
                kpt.append(mval)
                if mval > in_vis_thresh:
                    kpt_score += mval
                    count += 1

            if count > 0:
                kpt_score /= count
            rescore = kpt_score * score[idx].asscalar()

            self._results.append({'image_id': int(imgid[idx].asscalar()),
                                  'category_id': 1,
                                  'keypoints': kpt,
                                  'score': rescore})
            self._recorded_ids[int(imgid[idx].asscalar())] = True
