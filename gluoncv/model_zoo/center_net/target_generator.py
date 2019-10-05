"""CenterNet training target generator."""
from __future__ import absolute_import

import numpy as np

from mxnet import nd
from mxnet import gluon


class CenterNetTargetGenerator(gluon.Block):
    def __init__(self, num_class, output_width, output_height):
        super(CenterNetTargetGenerator, self).__init__()
        self._num_class = num_class
        self._output_width = output_width
        self._output_height = output_height

    def forward(self, im_width, im_height, gt_boxes, gt_ids):
        h_scale = self._output_height / im_height
        w_scale = self._output_width / im_width
        heatmap = np.zeros((num_classes, self._output_height, self._output_width), dtype=np.float32)
        wh_target = np.zeros((2, self._output_height, self._output_width), dtype=np.float32)
        wh_mask = np.zeros((2, self._output_height, self._output_width), dtype=np.float32)
        center_reg = np.zeros((2, self._output_height, self._output_width), dtype=np.float32)
        center_reg_mask = np.zeros((2, self._output_height, self._output_width), dtype=np.float32)
        for bbox, cid in zip(gt_boxes, gt_ids):
            box_h, box_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if box_h > 0 and box_w > 0:
                radius = _gaussian_radius((np.ceil(box_h), np.ceil(box_w)))
                radius = max(0, int(radius))
                center = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                center_int = center.astype(np.int32)
                center_x, center_y = center_int
                _draw_umich_gaussian(heatmap[cid], center_int, radius)
                wh_target[0, center_y, center_x] = box_w * w_scale
                wh_target[1, center_y, center_x] = box_h * h_scale
                wh_mask[:, center_y, center_x] = 1.0
                center_reg[:, center_y, center_x] = center - center_int
                center_reg_mask[:, center_y, center_x] = 1.0
        return tuple([nd.array(x) for x in (heatmap, wh_target, wh_mask, center_reg, center_reg_mask)])


def _gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)

def _draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]

  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
      np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap
