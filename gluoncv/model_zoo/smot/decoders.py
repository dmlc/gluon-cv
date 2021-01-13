"""
MXNet implementation of SMOT: Single-Shot Multi Object Tracking
https://arxiv.org/abs/2010.16031
"""
from mxnet import gluon
from gluoncv.nn.bbox import BBoxCenterToCorner


class NormalizedLandmarkCenterDecoder(gluon.HybridBlock):
    """
    Decode bounding boxes training target with normalized center offsets.
    This decoder must cooperate with NormalizedBoxCenterEncoder of same `stds`
    in order to get properly reconstructed bounding boxes.

    Returned bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.

    Parameters
    ----------
    stds : array-like of size 4
        Std value to be divided from encoded values, default is (0.1, 0.1, 0.2, 0.2).
    means : array-like of size 4
        Mean value to be subtracted from encoded values, default is (0., 0., 0., 0.).
    clip: float, default is None
        If given, bounding box target will be clipped to this value.

    """

    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.),
                 convert_anchor=True):
        super(NormalizedLandmarkCenterDecoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        self._stds = stds
        self._means = means
        if convert_anchor:
            self.center_to_conner = BBoxCenterToCorner(split=True)
        else:
            self.center_to_conner = None

    def hybrid_forward(self, F, x, anchors):
        """center decoder forward"""
        if self.center_to_conner is not None:
            a = self.center_to_conner(anchors)
        else:
            a = anchors.split(axis=-1, num_outputs=4)
        ld = F.split(x, axis=-1, num_outputs=10)

        x0 = F.broadcast_add(F.broadcast_mul(ld[0] * self._stds[0] + self._means[0], a[2] - a[0]), a[0])
        y0 = F.broadcast_add(F.broadcast_mul(ld[1] * self._stds[1] + self._means[1], a[3] - a[1]), a[1])
        x1 = F.broadcast_add(F.broadcast_mul(ld[2] * self._stds[0] + self._means[0], a[2] - a[0]), a[0])
        y1 = F.broadcast_add(F.broadcast_mul(ld[3] * self._stds[1] + self._means[1], a[3] - a[1]), a[1])
        x2 = F.broadcast_add(F.broadcast_mul(ld[4] * self._stds[0] + self._means[0], a[2] - a[0]), a[0])
        y2 = F.broadcast_add(F.broadcast_mul(ld[5] * self._stds[1] + self._means[1], a[3] - a[1]), a[1])
        x3 = F.broadcast_add(F.broadcast_mul(ld[6] * self._stds[0] + self._means[0], a[2] - a[0]), a[0])
        y3 = F.broadcast_add(F.broadcast_mul(ld[7] * self._stds[1] + self._means[1], a[3] - a[1]), a[1])
        x4 = F.broadcast_add(F.broadcast_mul(ld[8] * self._stds[0] + self._means[0], a[2] - a[0]), a[0])
        y4 = F.broadcast_add(F.broadcast_mul(ld[9] * self._stds[1] + self._means[1], a[3] - a[1]), a[1])

        return F.concat(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, dim=-1)


class GeneralNormalizedKeyPointsDecoder(gluon.HybridBlock):
    """
    Decode bounding boxes training target with normalized center offsets.
    This decoder must cooperate with NormalizedBoxCenterEncoder of same `stds`
    in order to get properly reconstructed bounding boxes.

    Returned bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.

    Parameters
    ----------
    stds : array-like of size 4
        Std value to be divided from encoded values, default is (0.1, 0.1, 0.2, 0.2).
    means : array-like of size 4
        Mean value to be subtracted from encoded values, default is (0., 0., 0., 0.).
    clip: float, default is None
        If given, bounding box target will be clipped to this value.

    """

    def __init__(self, num_points, stds=(0.2, 0.2), means=(0.5, 0.2),
                 convert_anchor=True):
        super(GeneralNormalizedKeyPointsDecoder, self).__init__()
        assert len(stds) == 2, "Box Encoder requires 4 std values."
        self._stds = stds
        self._means = means
        self._size = num_points * 2
        if convert_anchor:
            self.center_to_conner = BBoxCenterToCorner(split=True)
        else:
            self.center_to_conner = None

    def hybrid_forward(self, F, x, anchors):
        """key point decoder forward"""
        if self.center_to_conner is not None:
            a = self.center_to_conner(anchors)
        else:
            a = anchors.split(axis=-1, num_outputs=4)
        ld = F.split(x, axis=-1, num_outputs=self._size)

        outputs = []
        for i in range(0, self._size, 2):
            x = F.broadcast_add(F.broadcast_mul(ld[i] * self._stds[0] + self._means[0], a[2] - a[0]), a[0])
            y = F.broadcast_add(F.broadcast_mul(ld[i+1] * self._stds[1] + self._means[1], a[3] - a[1]), a[1])
            outputs.extend([x, y])

        return F.concat(*outputs, dim=-1)
