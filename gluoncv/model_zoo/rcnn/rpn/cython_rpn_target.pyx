import numpy as np

cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def sampler(np.ndarray[float, ndim=2] ious, float pos_iou_thresh,float neg_iou_thresh,
            int num_sample, int max_pos, float eps):
    cdef int num_anchors = ious.shape[0]
    cdef int num_gt = ious.shape[1]
    cdef int num_pos = 0
    cdef int i, j, num_neg, ind
    cdef float ious_max_per_anchor
    cdef np.ndarray[float, ndim=1] samples = np.zeros((num_anchors,), dtype=np.float32)
    cdef np.ndarray[int, ndim=1] matches = np.zeros((num_anchors,), dtype=np.int32)
    cdef np.ndarray[float, ndim=1] ious_max_per_gt = np.max(ious, axis=0)
    cdef np.ndarray[int, ndim=1] indexes = np.random.permutation(num_anchors).astype(np.int32)

    with nogil:
        for i in range(num_anchors):
            ious_max_per_anchor = -1.0
            for j in range(num_gt):
                if ious_max_per_anchor < ious[i, j]:
                    ious_max_per_anchor = ious[i, j]  # max
                    matches[i] = j  # argmax
            for j in range(num_gt):
                if ious[i, j] + eps > ious_max_per_gt[j]:
                    samples[i] = 1.0
                    num_pos += 1
                    break
            if ious_max_per_anchor >= pos_iou_thresh and samples[i] < 1.0:
                samples[i] = 1.0
                num_pos += 1
            if 0.0 <= ious_max_per_anchor < neg_iou_thresh:
                if samples[i] + eps > 1.0:
                    num_pos -=1
                samples[i] = -1.0
        num_pos = min(num_pos, max_pos)
        num_neg = num_sample - num_pos
        for i in range(num_anchors):
            ind = indexes[i]
            if samples[ind] > 0.0 and num_pos > 0:
                num_pos -= 1
            elif samples[ind] < 0.0 and num_neg > 0:
                num_neg -= 1
            elif samples[ind] > 0.0 and num_pos <= 0:
                samples[ind] = 0.0
            elif samples[ind] < 0.0 and num_neg <= 0:
                samples[ind] = 0.0
    return samples, matches