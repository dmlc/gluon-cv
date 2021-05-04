"""Modified NMS ops as static/dynamic layers"""
import numpy as np
import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat


__all__ = ['batched_nms', 'ml_nms']


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for iid in torch.unique(idxs).cpu().tolist():
        mask = (idxs == iid).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def ml_nms(boxlist, nms_thresh, max_proposals=-1, fixed_size=False):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Args:
        boxlist (data.structures.Boxes):
        nms_thresh (float): the score threshold for nms
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        fixed_size (bool): force output to be static shape
    """
    if nms_thresh <= 0:
        return boxlist
    boxes = boxlist.pred_boxes.tensor
    scores = boxlist.scores
    labels = boxlist.pred_classes
    keep = batched_nms(boxes, scores, labels, nms_thresh)
    # keep = nms(boxes, scores, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    if fixed_size:
        boxlist._keep = keep
        # the following code block is one way to perform a static nms op
        # scores = -scores
        # scores[keep] = -scores[keep]
        # result_mask = torch.zeros(scores.size()).to(scores.device)
        # result_mask[keep] = 1
        # scores = torch.where(result_mask == 1, scores, result_mask)
        boxlist.scores = scores
    else:
        boxlist = boxlist[keep]
    return boxlist

def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    """Calculate oks IOU score"""
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72,
                           .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious

def get_close_kpt_num(g, d, dis_thresh=1):
    """Calculate number of close keypoints"""
    xg = g[0::3]
    yg = g[1::3]
    close_keypoints_num = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        dx = xd - xg
        dy = yd - yg
        dis = np.sqrt(dx ** 2 + dy ** 2)
        close_keypoints_num[n_d] = (dis <= dis_thresh).sum()
    return close_keypoints_num

def oks_nms(boxlist, thresh, sigmas=None, in_vis_thre=None):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if len(boxlist) == 0:
        return boxlist

    kpts = boxlist.pred_keypoints.reshape(len(boxlist), -1).cpu().numpy()
    boxes = boxlist.pred_boxes.tensor
    areas = ((boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)).cpu().numpy()

    # scores = boxlist.scores
    # order = scores.argsort()[::-1]
    order = np.arange(0, len(boxlist))

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        inds = np.where(oks_ovr <= thresh)[0]
        order = order[inds + 1]

    return boxlist[keep]

def close_kpt_nms(boxlist, dis_thresh=1.0, close_kpt_thresh=5):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if len(boxlist) == 0:
        return boxlist

    kpts = boxlist.pred_keypoints.reshape(len(boxlist), -1).cpu().numpy()

    # scores = boxlist.scores
    # order = scores.argsort()[::-1]
    order = np.arange(0, len(boxlist))

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        close_kpt_num = get_close_kpt_num(kpts[i], kpts[order[1:]], dis_thresh)

        inds = np.where(close_kpt_num <= close_kpt_thresh)[0]
        order = order[inds + 1]

    return boxlist[keep]
