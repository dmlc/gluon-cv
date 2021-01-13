"""
MXNet implementation of tracktor in SMOT: Single-Shot Multi Object Tracking
https://arxiv.org/abs/2010.16031
"""
from abc import ABC, abstractmethod
import mxnet as mx
import numpy as np

from .utils import timeit
from .object_detector import JointObjectDetector


class BaseAnchorBasedTracktor(ABC):
    """Base tracktor class
    """

    @abstractmethod
    def anchors(self):
        """
        Returns the list of anchors used in this detector.
        -------

        """
        raise NotImplementedError

    @abstractmethod
    def prepare_for_frame(self, frame):
        """
        This method should run anything that needs to happen before the motion prediction.
        It can prepare the detector or even run the backbone feature extractions.
        It can also provide data to motion prediction

        Parameters
        ----------
            frame: the frame data, the same as in the detect_and_track method

        Returns
        -------
            motion_predict_data: optional data provided to motion prediction, if no data is provided, return None
        """
        raise NotImplementedError

    @abstractmethod
    def detect_and_track(self, frame, tracking_anchor_indices, tracking_anchor_weights, tracking_classes):
        """
        Perform detection and tracking on the new frame

        Parameters
        ----------
            frame: HxWx3 RGB image
            tracking_anchor_indices: NxM ndarray
            tracking_anchor_weights: NxM ndarray
            tracking_classes: Nx1 ndarray of the class ids of the tracked object

        Returns
            detection_bounding_boxes: all detection results, in (x0, y0, x1, y1, confidence, cls) format
            detection_source: source anchor box indices for each detection
            tracking_boxes: all tracking results, in (x0, y0, x1, y1, confidence) format
            extract_info: extra information from the tracktor, e.g. landmarks, a dict
        -------

        """
        raise NotImplementedError

    @abstractmethod
    def clean_up(self):
        """
        Clean up after running one video
        -------

        """
        raise NotImplementedError


class GluonSSDMultiClassTracktor(BaseAnchorBasedTracktor):
    """
    The tracktor based on GluonCV SSD detector.
    """
    def __init__(self, gpu_id, detector_thresh=0.5):

        self.detector = JointObjectDetector(gpu_id, data_shape=1080)
        self._anchor_tensor = None
        self._detector_thresh = detector_thresh
        self._ctx = mx.gpu(gpu_id)

        self._dummy_ti = mx.nd.array([[0]], ctx=self._ctx)

    def anchors(self):
        if self.detector.anchor_tensor is None:
            raise ValueError("anchor not initialized yet")
        return self.detector.anchor_tensor

    def clean_up(self):
        pass

    def prepare_for_frame(self, frame):
        return None

    @timeit
    #pylint: disable=arguments-differ
    def detect_and_track(self, frame, tracking_anchor_indices, tracking_anchor_weights, tracking_object_classes):

        with_tracking = len(tracking_anchor_indices) > 0

        if with_tracking:
            tracking_indices = mx.nd.array(tracking_anchor_indices, ctx=self._ctx)
            tracking_weights = mx.nd.array(tracking_anchor_weights, ctx=self._ctx)
            tracking_classes = mx.nd.array(tracking_object_classes.reshape((-1, 1)), ctx=self._ctx)
        else:
            tracking_classes = tracking_indices = tracking_weights = self._dummy_ti

        ids, scores, bboxes, voted_tracking_bboxes, detection_anchor_indices = \
            self.detector.run_detection(frame, tracking_indices, tracking_weights, tracking_classes)

        valid_det_num = (scores > self._detector_thresh).sum().astype(int).asnumpy()[0]

        if valid_det_num > 0:
            valid_scores = scores[:valid_det_num]
            valid_bboxes = bboxes[:valid_det_num, :]
            valid_classes = ids[:valid_det_num, :]
            detection_output = mx.nd.concat(valid_bboxes, valid_scores, valid_classes, dim=1).asnumpy()
            anchor_indices_output = detection_anchor_indices[:valid_det_num, :]
        else:
            # no detection
            detection_output = np.array([])
            anchor_indices_output = np.array([])

        tracking_response = voted_tracking_bboxes.asnumpy() \
            if with_tracking else np.array([])

        return detection_output, anchor_indices_output, tracking_response, {}
