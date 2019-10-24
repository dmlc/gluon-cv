import numpy as np

cimport numpy as np
cimport cython

from libc.math cimport log

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def np_normalized_box_encoder(np.ndarray[float, ndim=2] samples, np.ndarray[long, ndim=2] matches,
                              np.ndarray[float, ndim=3] anchors, np.ndarray[float, ndim=3] refs,
                              np.ndarray[float, ndim=1] means, np.ndarray[float, ndim=1] stds):
    cdef int b = refs.shape[0]
    cdef int n = anchors.shape[1]
    cdef float ref_width, ref_height, ref_x, ref_y, ref_xmin, ref_ymin
    cdef float a_width, a_height, a_x, a_y, a_xmin, a_ymin
    cdef float t0, t1, t2, t3
    cdef int i, j, k, match
    cdef np.ndarray[float, ndim=3] targets = np.zeros((b, n, 4), dtype=np.float32)
    cdef np.ndarray[float, ndim=3] masks = np.zeros((b, n, 4), dtype=np.float32)
    with nogil:
        for i in range(b):
            for j in range(n):
                match = matches[i, j]
                # xmin: 0, ymin:1, xmax: 2, ymax: 3
                # x:0, y:1, w:2, h:3
                ref_xmin = refs[i, match, 0]
                ref_ymin = refs[i, match, 1]
                ref_width = refs[i, match, 2] - ref_xmin
                ref_height = refs[i, match, 3] - ref_ymin
                ref_x = ref_xmin + ref_width * 0.5
                ref_y = ref_ymin + ref_height * 0.5
                a_xmin = anchors[i, j, 0]
                a_ymin = anchors[i, j, 1]
                a_width = anchors[i, j, 2] - a_xmin
                a_height = anchors[i, j, 3] - a_ymin
                a_x = a_xmin + a_width * 0.5
                a_y = a_ymin + a_height * 0.5
                t0 = ((ref_x - a_x) / a_width - means[0]) / stds[0]
                t1 = ((ref_y - a_y) / a_height - means[1]) / stds[1]
                t2 = (log(ref_width / a_width) - means[2]) / stds[2]
                t3 = (log(ref_height / a_height) - means[3]) / stds[3]
                valid = 1 if samples[i, j] > 0.5 else 0
                masks[i, j, 0] = valid
                masks[i, j, 1] = valid
                masks[i, j, 2] = valid
                masks[i, j, 3] = valid
                targets[i, j, 0] = t0 if valid else 0.0
                targets[i, j, 1] = t1 if valid else 0.0
                targets[i, j, 2] = t2 if valid else 0.0
                targets[i, j, 3] = t3 if valid else 0.0
    return targets, masks
