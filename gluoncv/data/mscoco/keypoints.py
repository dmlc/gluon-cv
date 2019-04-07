"""MS COCO keypoints dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import copy
import numpy as np
import mxnet as mx
from .utils import try_import_pycocotools
from ..base import VisionDataset
from ...utils.bbox import bbox_xywh_to_xyxy, bbox_clip_xyxy


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
    CLASSES = ['person']

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'coco'),
                 splits=('person_keypoints_val2017',), aspect_ratio=1.0, skip_empty=True):
        super(COCOKeyPoints, self).__init__(root)
        self._root = os.path.expanduser(root)
        if isinstance(splits, mx.base.string_types):
            splits = [splits]
        self._splits = splits
        self._coco = []
        self._aspect_ratio = float(1.0/aspect_ratio)
        self._pixel_std = 200
        self._skip_empty = skip_empty
        self.index_map = dict(zip(type(self).CLASSES, range(self.num_class)))
        self.json_id_to_contiguous = None
        self.contiguous_id_to_json = None
        self._items, self._labels = self._load_jsons()

        # properties may help
        '''
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
            [16,14], [14,12], [17,15], [15,13], [12,13], [6,12], [7,13], [6,7], [6,8],
            [7,9], [8,10], [9,11], [2,3], [1,2], [1,3], [2,4], [3,5], [4,6], [5,7]]
        '''

    def __str__(self):
        detail = ','.join([str(s) for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

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
        if len(self._coco) > 1:
            raise NotImplementedError(
                "Currently we don't support evaluating {} JSON files".format(len(self._coco)))
        return self._coco[0]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_path = self._items[idx]
        img_id = int(os.path.splitext(os.path.basename(img_path))[0])

        label = copy.deepcopy(self._labels[idx])
        img = mx.image.imread(img_path, 1)
        return img, label, img_id

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []
        # lazy import pycocotools
        try_import_pycocotools()
        from pycocotools.coco import COCO
        for split in self._splits:
            anno = os.path.join(self._root, 'annotations', split) + '.json'
            _coco = COCO(anno)
            self._coco.append(_coco)
            classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
            if not classes == self.classes:
                raise ValueError("Incompatible category names with COCO: ")
            assert classes == self.classes
            json_id_to_contiguous = {
                v: k for k, v in enumerate(_coco.getCatIds())}
            if self.json_id_to_contiguous is None:
                self.json_id_to_contiguous = json_id_to_contiguous
                self.contiguous_id_to_json = {
                    v: k for k, v in self.json_id_to_contiguous.items()}
            else:
                assert self.json_id_to_contiguous == json_id_to_contiguous
            # iterate through the annotations
            image_ids = sorted(_coco.getImgIds())
            for entry in _coco.loadImgs(image_ids):
                dirname, filename = entry['coco_url'].split('/')[-2:]
                abs_path = os.path.join(self._root, dirname, filename)
                if not os.path.exists(abs_path):
                    raise IOError('Image: {} not exists.'.format(abs_path))
                label = self._check_load_keypoints(_coco, entry)
                if not label:
                    continue

                # num of items are relative to person, not image
                for obj in label:
                    items.append(abs_path)
                    labels.append(obj)
        return items, labels

    def _check_load_keypoints(self, coco, entry):
        """Check and load ground-truth keypoints"""
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=False)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']

        for obj in objs:
            contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
            if contiguous_cid >= self.num_class:
                # not class of interest
                continue
            if max(obj['keypoints']) == 0:
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            if obj['area'] <= 0 or xmax <= xmin or ymax <= ymin:
                continue

            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]
                joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]
                # joints_3d[i, 2, 0] = 0
                visible = min(1, obj['keypoints'][i * 3 + 2])
                joints_3d[i, :2, 1] = visible
                # joints_3d[i, 2, 1] = 0
                center, scale = self._box_to_center_scale(xmin, ymin, xmax - xmin, ymax - ymin)

            valid_objs.append({
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d
            })

        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'center': np.array([-1., -1.]),
                    'scale': np.array([0., 0.]),
                    'joints_3d': np.zeros((self.num_joints, 3, 2), dtype=np.float32)
                })
        return valid_objs

    def _box_to_center_scale(self, x, y, w, h):
        """Convert box coordinates to center and scale.
        adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
        """
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self._aspect_ratio * h:
            h = w / self._aspect_ratio
        elif w < self._aspect_ratio * h:
            w = h * self._aspect_ratio
        scale = np.array(
            [w * 1.0 / self._pixel_std, h * 1.0 / self._pixel_std], dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25
        return center, scale
