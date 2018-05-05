"""Faster RCNN BBox Utils"""
import mxnet as mx
from mxnet import nd

def bbox_transform(anchor, bbox):
    w = anchor[:, 2] - anchor[:, 0]
    h = anchor[:, 3] - anchor[:, 1]
    cx = (anchor[:, 0] + anchor[:, 2]) / 2.0
    cy = (anchor[:, 1] + anchor[:, 3]) / 2.0

    
    g_w = bbox[:, 2] - bbox[:, 0]
    g_h = bbox[:, 3] - bbox[:, 1]
    g_cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
    g_cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
    
    g_w = mx.ndarray.log(g_w / w)
    g_h = mx.ndarray.log(g_h / h)
    g_cx = (g_cx - cx) / w 
    g_cy = (g_cy - cy) / h
    return mx.ndarray.concatenate([
                g_w.reshape((-1, 1)), 
                g_h.reshape((-1, 1)), 
                g_cx.reshape((-1, 1)), 
                g_cy.reshape((-1, 1))], axis=1)


def bbox_inverse_transform(anchor, bbox):
    w = anchor[:, 2] - anchor[:, 0]
    h = anchor[:, 3] - anchor[:, 1]
    cx = (anchor[:, 0] + anchor[:, 2]) / 2.0
    cy = (anchor[:, 1] + anchor[:, 3]) / 2.0

    g_w = mx.ndarray.exp(bbox[:, 0]) * w
    g_h = mx.ndarray.exp(bbox[:, 1]) * h
    g_cx = bbox[:, 2] * w + cx
    g_cy = bbox[:, 3] * h + cy

    g_x1 = g_cx - g_w / 2
    g_y1 = g_cy - g_h / 2
    g_x2 = g_cx + g_w / 2
    g_y2 = g_cy + g_h / 2
    return mx.ndarray.concatenate([
                        g_x1.reshape((-1, 1)),
                        g_y1.reshape((-1, 1)),
                        g_x2.reshape((-1, 1)),
                        g_y2.reshape((-1, 1))], axis=1)


def bbox_clip(bbox:mx.nd.NDArray, height, width):
    zeros_t = mx.nd.zeros(bbox[:, 0].shape, ctx=bbox.context)
    bbox[:, 0] = mx.nd.maximum(bbox[:, 0], zeros_t) 
    bbox[:, 1] = mx.nd.maximum(bbox[:, 1], zeros_t) 
    bbox[:, 2] = mx.nd.minimum(bbox[:, 2], zeros_t + width - 1) 
    bbox[:, 3] = mx.nd.minimum(bbox[:, 3], zeros_t + height - 1) 
    return bbox
