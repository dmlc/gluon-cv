"""Evaluation Metrics for Semantic Segmentation"""
import threading
import numpy as np
import mxnet as mx
import mxnet.ndarray as F
from mxnet.metric import EvalMetric

__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union',
           'pixelAccuracy', 'intersectionAndUnion']

class SegmentationMetric(EvalMetric):
    """Computes pixAcc and mIoU metric scroes
    """
    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__('pixAcc & mIoU')
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            correct, labeled = batch_pix_accuracy(
                pred, label)
            inter, union = batch_intersection_union(
                pred, label, self.nclass)
            with self.lock:
                self.total_correct += correct
                self.total_label += labeled
                self.total_inter += inter
                self.total_union += union
            return

        if isinstance(preds, mx.nd.NDArray):
            evaluate_worker(self, labels, preds)
            return
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                       )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            return

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU
 
    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        return

def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are NDarray, output 4D, target 3D
    # the category -1 is ignored class, typically for background / boundary
    predict = F.argmax(output, 1)
    predict = predict.asnumpy() + 1
    target = target.asnumpy().astype(predict.dtype) + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target)*(target > 0))
    """
    predict = np.argmax(output.asnumpy(), 1) + 1
    target = target + 1
    pixel_labeled = (target > 0).sum().asnumpy()[0]
    pixel_correct = (F.equal(predict, target)*(target > 0)).sum().asnumpy()[0]
    """
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are NDarray, output 4D, target 3D
    # the category -1 is ignored class, typically for background / boundary
    predict = F.argmax(output, 1)
    target = target.astype(predict.dtype)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.asnumpy() + 1
    target = target.asnumpy() + 1

    predict = predict * (target > 0).astype(predict.dtype)
    #intersection = predict * (F.equal(predict, target)).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Unition's"
    return area_inter, area_union


def pixelAccuracy(imPred, imLab):
    """
    This function takes the prediction and label of a single image, returns pixel-wise accuracy
    To compute over many images do:
    for i = range(Nimages):
         (pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = \
            pixelAccuracy(imPred[i], imLab[i])
    mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab > 0)
    pixel_correct = np.sum((imPred == imLab)*(imLab > 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return (pixel_accuracy, pixel_correct, pixel_labeled)


def intersectionAndUnion(imPred, imLab, numClass):
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)
