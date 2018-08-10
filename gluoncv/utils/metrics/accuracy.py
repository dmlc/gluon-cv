"""Accuracy metirc with ignored labels."""
# pylint: disable=assignment-from-no-return
import numpy as np
import mxnet as mx
from mxnet import ndarray
from mxnet.metric import check_label_shapes


class Accuracy(mx.metric.EvalMetric):
    """Computes accuracy classification score with optional ignored labels.
    The accuracy score is defined as
    .. math::
        \\text{accuracy}(y, \\hat{y}) = \\frac{1}{n} \\sum_{i=0}^{n-1}
        \\text{1}(\\hat{y_i} == y_i)
    Parameters
    ----------
    axis : int, default=1
        The axis that represents classes
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    ignore_labels : int or iterable of integers, optional
        If provided as not None, will ignore these labels during update.
    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> acc = mx.metric.Accuracy()
    >>> acc.update(preds = predicts, labels = labels)
    >>> print acc.get()
    ('accuracy', 0.6666666666666666)
    """
    def __init__(self, axis=1, name='accuracy',
                 output_names=None, label_names=None, ignore_labels=None):
        super(Accuracy, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names)
        self.axis = axis
        self.ignore_labels = np.array(ignore_labels).flatten()

    def update(self, labels, preds):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data with class indices as values, one per sample.
        preds : list of `NDArray`
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')

            labels, preds = check_label_shapes(label, pred_label)

            valid = (labels.reshape(-1, 1) != self.ignore_labels).all(axis=-1)

            self.sum_metric += np.logical_and(pred_label.flat == label.flat, valid).sum()
            self.num_inst += np.sum(valid)
