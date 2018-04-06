import numpy as np
import mxnet as mx
import mxnet.ndarray as F

def batch_pix_accuracy(output, target, bg=False):
    # inputs are NDarray, output 4D, target 3D
    predict = F.argmax(output, 1)
    if bg:
        pixel_labeled = (target > 0).sum()
        pixel_correct = (F.equal(predict, target)*(target>0)).sum().asnumpy()[0]
    else:
        pixel_labeled = target.size
        pixel_correct = F.equal(predict.astype(target.dtype), target).sum().asnumpy()[0]
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass, bg=False):
    # inputs are NDarray, output 4D, target 3D
    predict = F.argmax(output, 1)
    mini = 0
    nbins = nclass
    maxi = nclass - 1
    if bg:
        mini = 1
        nbins -= 1
        predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (F.equal(predict, target.astype(predict.dtype))).astype(predict.dtype)
    # bg=True, ignoring class 0; bg = False, use class 0
    # areas of intersection and union 
    area_inter,_ = np.histogram(intersection.asnumpy(), bins=nbins,
                                range=(mini, maxi))
    area_pred,_ = np.histogram(predict.asnumpy(), bins=nbins,
                               range=(mini, maxi))
    area_lab,_ = np.histogram(target.asnumpy(), bins=nbins,
                              range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union
