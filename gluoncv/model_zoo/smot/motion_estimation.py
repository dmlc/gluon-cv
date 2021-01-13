"""
Motion estimation module for MOT
"""
# pylint: disable=unused-argument,arguments-differ,missing-class-docstring
import time
import logging
from abc import ABC, abstractmethod
import numpy as np


class MotionEstimator(ABC):

    @abstractmethod
    def initialize(self, first_frame, first_frame_motion_pred_data):
        """
        Initialize the motion estimator by feeding the first frame

        Parameters
        ----------
        first_frame: data of the first frame
        first_frame_motion_pred_data: additional data for motion prediction

        Returns:
            cache_information
        -------

        """

    @abstractmethod
    def predict_new_locations(self,
                              prev_frame_cache: np.ndarray,
                              prev_bboxes: np.ndarray,
                              new_frame: np.ndarray,
                              new_frame_motion_pred_data,
                              tracked_boxes_anchor_indices=None,
                              tracked_boxes_anchor_weights=None,
                              skip: bool = False,
                              **kwargs):
        """
        The abstract method for predicting movement of bounding boxes given the two frames.
        Parameters
        ----------
        prev_frame_cache: cached image from motion estimation, numpy.ndarray
        prev_bboxes: Nx4 numpy.ndarray, bounding boxes in (left, top, right, bottom) format
        new_frame: BGR image, numpy.ndarray
        new_frame_motion_pred_data: additional data for motion prediction
        tracked_boxes_anchor_indices: anchor indices used to build the prev_bboxes
        tracked_boxes_anchor_weights: voting weights of anchors used to build prev_bboxes
        skip: whether to just skip motion estimation for this frame
        kwargs: other information
        Returns
            new_boxes: Nx4 numpy.ndarray
            cache_information:
        -------

        """
        # pylint: disable=notimplemented-raised,raising-bad-type
        raise NotImplemented


class DummyMotionEstimator(MotionEstimator):

    def initialize(self, first_frame, first_frame_motion_pred_data):
        pass

    def predict_new_locations(self,
                              prev_frame_cache: np.ndarray,
                              prev_bboxes: np.ndarray,
                              new_frame: np.ndarray,
                              skip: bool = False,
                              **kwargs):
        return prev_bboxes, None


class BaseFlowMotionEstimator(MotionEstimator):
    """
    The basic structure of a flow based motion estimator
    To implement your own flow tracker, extend the followting methods:

    compuate_flow(): given preprocessed information, compute optical flow map
    prepare_frame(): for preprocessing, the results will be stored and provided to next frame's inference
    """

    def initialize(self, first_frame, first_frame_motion_pred_data):
        prepared_first_frame = self.prepare_frame(first_frame)
        return prepared_first_frame

    def predict_new_locations(self,
                              prev_frame_cache: np.ndarray,
                              prev_bboxes: np.ndarray,
                              new_frame: np.ndarray,
                              new_frame_motion_pred_data,
                              skip: bool = False,
                              **kwargs):

        t_start_prepare = time.time()
        prepared_new_frame = self.prepare_frame(new_frame)
        assert prev_frame_cache.shape == prepared_new_frame.shape
        e_prepare = time.time() - t_start_prepare
        logging.info("flow preparation runtime: {:.05f}".format(e_prepare))

        if not skip:
            t_start_flow = time.time()
            flow_map = self.compute_flow(prev_frame_cache, prepared_new_frame)
            assert flow_map.shape[-1] == 2, ValueError("flow map elements must be 2 element vectors!")
            e_flow = time.time() - t_start_flow
            logging.info("flow computation runtime: {:.05f}".format(e_flow))

            t_start_pred = time.time()
            predicted_bboxes = self._warp_bbox(new_frame.shape, prev_bboxes, flow_map)
            e_pred = time.time() - t_start_pred
            logging.info("bounding box prediction runtime: {:.05f}".format(e_pred))
            return predicted_bboxes, prepared_new_frame
        else:
            return prev_bboxes, prepared_new_frame

    @abstractmethod
    def compute_flow(self, prev_frame_cache, prepared_new_frame):
        """
        Compute dense optical flow
        Parameters
        ----------
        prev_frame_cache
        prepared_new_frame

        Returns
            flow_map: a NxMx2 map. each spatial local contains a 2-element vector
            specifying the delta in x and y directions. The unit of delta is pixel
            in this flow_map's coordinate space
        -------

        """
        # pylint: disable=notimplemented-raised,raising-bad-type
        raise NotImplementedError

    @abstractmethod
    def prepare_frame(self, frame):
        # pylint: disable=notimplemented-raised,raising-bad-type
        raise not NotImplementedError

    @classmethod
    def _warp_bbox(cls, image_shape: tuple, bboxes: np.ndarray, flow_map: np.ndarray):
        """
        Use the computed denseflow map to estimate the movement of the object bounding box
        then warp the boxes to their new locations.

        Parameters
        ----------
        image_shape: tuple
        bboxes: numpy.ndarray shape: (n, 4)
        flow_map: numpy.ndarray shape: (h, w, 2)

        Returns: predicted new locations of the boxes, numpy.ndarray shape: (n, 4)
        -------

        """
        # pylint: disable=logging-format-interpolation,bare-except
        flow_shape = flow_map.shape

        flow_h, flow_w, _ = flow_shape

        t0 = time.time()
        image_h = image_shape[0]
        image_w = image_shape[1]

        image2flow_ratio_h = image_h / flow_h
        image2flow_ratio_w = image_w / flow_w

        logging.info("ratio: {}x{}".format(image2flow_ratio_w, image2flow_ratio_h))

        # Remap the delta to image coordinate space
        flow_map[..., 0] *= image2flow_ratio_w
        flow_map[..., 1] *= image2flow_ratio_h

        flow_map_bboxes = bboxes.copy()
        flow_map_bboxes[:, 0] /= image2flow_ratio_w
        flow_map_bboxes[:, 2] /= image2flow_ratio_w
        flow_map_bboxes[:, 1] /= image2flow_ratio_h
        flow_map_bboxes[:, 3] /= image2flow_ratio_h
        warped_bboxes = []

        for flow_bbox, raw_bbox in zip(flow_map_bboxes, bboxes):

            x0, y0, x1, y1 = int(flow_bbox[0]), int(flow_bbox[1]), int(flow_bbox[2]), int(flow_bbox[3])

            delta_x = flow_map[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1), 0]
            delta_y = flow_map[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1), 1]

            dx = np.mean(delta_x)
            dy = np.mean(delta_y)

            if np.isnan(dx) or np.isnan(dy):
                warped_bboxes.append(raw_bbox)
            else:
                warped_bboxes.append([raw_bbox[0] + dx, raw_bbox[1] + dy, raw_bbox[2] + dx, raw_bbox[3] + dy])
        logging.info("bbox warping runtime cost {}".format(time.time() - t0))

        return np.array(warped_bboxes)


class FarneBeckFlowMotionEstimator(BaseFlowMotionEstimator):
    """
    Use the farnebeck algorithm for the flow-based motion estimator
    """

    def __init__(self, flow_scale=256):
        self.flow_scale = flow_scale
        import cv2
        self._cv2 = cv2

    def prepare_frame(self, frame):
        img_h, img_w, _ = frame.shape
        ratio = min(self.flow_scale / img_w, self.flow_scale / img_h)
        flow_h, flow_w = int(img_h * ratio), int(img_w * ratio)
        cv2 = self._cv2
        resized_gray_frame = cv2.cvtColor(
            cv2.resize(frame, (flow_w, flow_h), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_BGR2GRAY)
        return resized_gray_frame

    def compute_flow(self, prev_frame_cache, prepared_new_frame):
        # Compute Farnebeck flow
        cv2 = self._cv2
        flow_map = cv2.calcOpticalFlowFarneback(
            prev_frame_cache, prepared_new_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        return flow_map
