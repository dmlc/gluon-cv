"""MatrixNet training target generator."""
from __future__ import absolute_import

import numpy as np

from mxnet import nd
from mxnet import gluon

def layer_map_using_ranges(width, height, layer_ranges, fpn_flag=0):
    layers = []
   
    for i, layer_range in enumerate(layer_ranges):
        if fpn_flag ==0:
            if (width >= 0.8 * layer_range[2]) and (width <= 1.3 * layer_range[3]) and (height >= 0.8 * layer_range[0]) and (height <= 1.3 * layer_range[1]):
                layers.append(i)
        else:
            max_dim = max(height, width)
            if max_dim <= 1.3*layer_range[1] and max_dim >= 0.8* layer_range[0]:
                layers.append(i)
    if len(layers) > 0:
        return layers
    else:
        return [len(layer_ranges) - 1]


class MatrixNetTargetGenerator(gluon.Block):
    """Target generator for CenterNet.

    Parameters
    ----------
    num_class : int
        Number of categories.
    output_width : int
        Width of the network output.
    output_height : int
        Height of the network output.

    """
    def __init__(self, num_class, input_width, input_height, layers_range):
        super(MatrixNetTargetGenerator, self).__init__()
        self._num_class = num_class
        self._input_width = int(input_width)
        self._input_height = int(input_height)
        self._layers_range = layers_range

    def forward(self, gt_boxes, gt_ids):
        """Target generation"""
        # pylint: disable=arguments-differ
        _dict={}
        output_sizes=[]
        # indexing layer map
        for i,l in enumerate(self._layers_range):
            for j,e in enumerate(l):
                if e !=-1:
                    output_sizes.append([self._input_height//(8*2**(j)), self._input_width//(8*2**(i))])
                    _dict[(i+1)*10+(j+1)]=e
    
        self._layers_range=[_dict[i] for i in sorted(_dict)]
        fpn_flag = set(_dict.keys()) == set([11,22,33,44,55])
        
        heatmaps = [np.zeros((self._num_class, output_size[0], output_size[1]), dtype=np.float32) for output_size in output_sizes]
        wh_targets = [np.zeros((2, output_size[0], output_size[1]), dtype=np.float32) for output_size in output_sizes]
        wh_masks = [np.zeros((2, output_size[0], output_size[1]), dtype=np.float32) for output_size in output_sizes]
        center_regs = [np.zeros((2, output_size[0], output_size[1]), dtype=np.float32) for output_size in output_sizes]
        center_reg_masks = [np.zeros((2, output_size[0], output_size[1]), dtype=np.float32) for output_size in output_sizes]
        for bbox, cid in zip(gt_boxes, gt_ids):
            for olayer_idx in layer_map_using_ranges(bbox[2] - bbox[0], bbox[3] - bbox[1], self._layers_range, fpn_flag):
                cid = int(cid)
                width_ratio = output_sizes[olayer_idx][1] / self._input_width
                height_ratio = output_sizes[olayer_idx][0] / self._input_height
                xtl, ytl = bbox[0], bbox[1]
                xbr, ybr = bbox[2], bbox[3]
                fxtl = (xtl * width_ratio)
                fytl = (ytl * height_ratio)
                fxbr = (xbr * width_ratio)
                fybr = (ybr * height_ratio)
                                
                box_h, box_w = fybr - fytl, fxbr - fxtl
                if box_h > 0 and box_w > 0:
                    radius = _gaussian_radius((np.ceil(box_h), np.ceil(box_w)))
                    radius = max(0, int(radius))
                    center = np.array(
                        [(fxtl + fxbr) / 2 , (fytl + fybr) / 2 ],
                        dtype=np.float32)
                    center_int = center.astype(np.int32)
                    center_x, center_y = center_int
                    assert center_x < output_sizes[olayer_idx][1], \
                        'center_x: {} > output_width: {}'.format(center_x, output_sizes[olayer_idx][1])
                    assert center_y < output_sizes[olayer_idx][0], \
                        'center_y: {} > output_height: {}'.format(center_y, output_sizes[olayer_idx][0])
                    _draw_umich_gaussian(heatmaps[olayer_idx][cid], center_int, radius)
                    wh_targets[olayer_idx][0, center_y, center_x] = box_w 
                    wh_targets[olayer_idx][1, center_y, center_x] = box_h 
                    wh_masks[olayer_idx][:, center_y, center_x] = 1.0
                    center_regs[olayer_idx][:, center_y, center_x] = center - center_int
                    center_reg_masks[olayer_idx][:, center_y, center_x] = 1.0
        heatmaps = [nd.array(heatmap) for heatmap in heatmaps]
        wh_targets = [nd.array(wh_target) for wh_target in wh_targets]
        wh_masks = [nd.array(wh_mask) for wh_mask in wh_masks]
        center_regs = [nd.array(center_reg) for center_reg in center_regs]
        center_reg_masks = [nd.array(center_reg_mask) for center_reg_mask in center_reg_masks]
        return heatmaps, wh_targets, wh_masks, center_regs, center_reg_masks


def _gaussian_radius(det_size, min_overlap=0.7):
    """Calculate gaussian radius for foreground objects.

    Parameters
    ----------
    det_size : tuple of int
        Object size (h, w).
    min_overlap : float
        Minimal overlap between objects.

    Returns
    -------
    float
        Gaussian radius.

    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def _gaussian_2d(shape, sigma=1):
    """Generate 2d gaussian.

    Parameters
    ----------
    shape : tuple of int
        The shape of the gaussian.
    sigma : float
        Sigma for gaussian.

    Returns
    -------
    float
        2D gaussian kernel.

    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def _draw_umich_gaussian(heatmap, center, radius, k=1):
    """Draw a 2D gaussian heatmap.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap to be write inplace.
    center : tuple of int
        Center of object (h, w).
    radius : type
        The radius of gaussian.

    Returns
    -------
    numpy.ndarray
        Drawn gaussian heatmap.

    """
    diameter = 2 * radius + 1
    gaussian = _gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
