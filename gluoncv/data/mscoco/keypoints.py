"""MS COCO keypoints dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import numpy as np
import mxnet as mx
from .utils import try_import_pycocotools
from ..base import VisionDataset


class COCOKeyPoints(VisionDataset):
    """COCO keypoint detection dataset.

    Parameters
    ----------
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of str, default ['person_keypoints_val2017']
        Json annotations name.
        Candidates can be: person_keypoints_val2017, person_keypoints_train2017.


    """
    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'coco'),
                 splits=('person_keypoints_val2017',)):
        super(COCOKeyPoints, self).__init__(root)
        self._root = os.path.expanduser(root)
        if isinstance(splits, mx.base.string_types):
            splits = [splits]
        self._splits = splits
        self._coco = []
        self._items, self._labels = self._load_jsons()

        # properties may help
        self._keypoints: {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "left_shoulder",
            6: "right_shoulder",
            7: "left_elbow",
            8: "right_elbow",
            9: "left_wrist",
            10: "right_wrist",
            11: "left_hip",
            12: "right_hip",
            13: "left_knee",
            14: "right_knee",
            15: "left_ankle",
            16: "right_ankle"
        }

        self._skeleton: [
            [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
            [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]

    def __str__(self):
        detail = ','.join([str(s) for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def num_joints(self):
        """Dataset defined: number of joints provided."""
        return 17

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]

    @property
    def coco(self):
        """Return pycocotools object for evaluation purposes."""
        if not self._coco:
            raise ValueError("No coco objects found, dataset not initialized.")
        elif len(self._coco) > 1:
            raise NotImplementedError(
                "Currently we don't support evaluating {} JSON files".format(len(self._coco)))
        return self._coco[0]

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []
        # lazy import pycocotools
        try_import_pycocotools()
        from pycocotools.coco import COCO
