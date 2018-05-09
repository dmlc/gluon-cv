from __future__ import print_function

import numpy as np
import gluoncv as gcv

def test_bbox_xywh_to_xyxy():
    # test list
    a = [20, 30, 100.2, 300.4]
    expected = [20, 30, 119.2, 329.4]
    np.testing.assert_allclose(gcv.utils.bbox.bbox_xywh_to_xyxy(a), expected)
    aa = np.array([a, a])
    bb = np.array([expected, expected])
    np.testing.assert_allclose(gcv.utils.bbox.bbox_xywh_to_xyxy(aa), bb)

if __name__ == '__main__':
    import nose
    nose.runmodule()
