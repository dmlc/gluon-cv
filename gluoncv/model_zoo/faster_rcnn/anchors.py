# The code is adapted from the original Faster R-CNN Python implementation.
import mxnet.ndarray as nd

def generate_anchors(base_size=16, ratios=nd.array([0.5, 1, 2]), scales=nd.array([8, 16, 32])):
    """
    Generate anchor (reference) windows by enumerating aspect ratios and scales
    wrt a reference (0, 0, 15, 15) window.
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/generate_anchors.py
    """
    base_anchor = nd.array([1, 1, base_size, base_size])
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = nd.concatenate([_scale_enum(ratio_anchors[i, :], scales)
                              for i in range(ratio_anchors.shape[0])])
    return anchors

def map_anchors(ref_anchors, feat_shape, img_h, img_w, ctx):
    """Map the anchors based on the image size
    """
    C, H, W = feat_shape[1:]
    # broadcast anchors to batch and locations
    ref_anchors = ref_anchors.as_in_context(ctx)
    ref_anchors = ref_anchors.reshape((1, -1, 1, 1))
    ref_anchors = ref_anchors.broadcast_to(feat_shape)
    # img_w * (0.0: 1/w: 1.0)
    ref_x = nd.arange(W).as_in_context(ctx).reshape((1, W)) / W
    ref_x = ref_x * img_w
    ref_x = ref_x.broadcast_to((H, W))
    # img_h * (0.0: 1/h: 1.0)
    ref_y = nd.arange(H).as_in_context(ctx).reshape((H, 1)) / H
    ref_y = ref_y * img_h
    ref_y = ref_y.broadcast_to((H, W))
    for anchor_i in range(C//4):
        ref_anchors[:, anchor_i * 4] += ref_x
        ref_anchors[:, anchor_i * 4 + 1] += ref_y
        ref_anchors[:, anchor_i * 4 + 2] += ref_x
        ref_anchors[:, anchor_i * 4 + 3] += ref_y
    return ref_anchors

def _whctrs(anchor:nd.NDArray):
    """Return width, height, x center, and y center for an anchor (window).
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws.reshape((-1, 1))
    hs = hs.reshape((-1, 1))
    anchors = nd.concatenate(
                        [x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)], axis=1)
    return anchors

def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = nd.round(nd.sqrt(size_ratios))
    hs = nd.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
