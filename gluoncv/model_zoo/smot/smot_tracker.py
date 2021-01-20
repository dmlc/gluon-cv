"""
MXNet implementation of tracktor in SMOT: Single-Shot Multi Object Tracking
https://arxiv.org/abs/2010.16031
"""
# pylint: disable=line-too-long,logging-format-interpolation,unused-argument,missing-function-docstring
from __future__ import absolute_import
import logging
import time
import numpy as np

import mxnet as mx
from gluoncv.utils.bbox import bbox_iou
from .utils import timeit, Track
from .motion_estimation import FarneBeckFlowMotionEstimator
from .motion_estimation import DummyMotionEstimator


def nms_fallback(boxes, thresh):
    """
    Perform non-maximal suppression and return the indices
    Parameters
    ----------
    boxes: [[x, y, xmax, ymax, score]]

    Returns kept box indices
    -------

    """
    order = np.argsort(boxes[:, -1])[::-1]

    iou_mat = bbox_iou(boxes[:, :4], boxes[:, :4])

    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        IOU = iou_mat[i, order[1:]]

        remaining = np.where(IOU <= thresh)[0]
        order = order[remaining + 1]

    return keep


def gpu_iou(bbox_a_tensor, bbox_b_tensor):
    """

    Parameters
    ----------
    bbox_a_tensor
    bbox_b_tensor

    Returns
    -------

    """
    if bbox_a_tensor.shape[1] < 4 or bbox_b_tensor.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = mx.nd.maximum(bbox_a_tensor.reshape((bbox_a_tensor.shape[0], 1, -1))[:, :, :2], bbox_b_tensor[:, :2])
    br = mx.nd.minimum(bbox_a_tensor.reshape((bbox_a_tensor.shape[0], 1, -1))[:, :, 2:4], bbox_b_tensor[:, 2:4])

    valid = mx.nd.prod(tl < br, axis=2)
    area_i = mx.nd.prod(br - tl, axis=2) * valid
    area_a = mx.nd.prod(bbox_a_tensor[:, 2:4] - bbox_a_tensor[:, :2], axis=1)
    area_b = mx.nd.prod(bbox_b_tensor[:, 2:4] - bbox_b_tensor[:, :2], axis=1)
    return area_i / (area_a.expand_dims(1) + area_b.expand_dims(0) - area_i)


class SMOTTracker:
    """
    Implementation of the SMOT tracker
    The steps to use the tracker is:
    0. Set anchors from the SSD
    1. First call tracker.predict(new_frame)
    2. Then get the tracking anchor information
    3. Run the detractor with the tracking anchor information
    4. Run tracker.update(new_detection, track_info).
    """

    def __init__(self,
                 motion_model='no',
                 anchor_array=None,
                 use_motion=True,
                 match_top_k=10,
                 track_keep_alive_thresh=0.1,
                 new_track_iou_thresh=0.3,
                 track_nms_thresh=0.5,
                 gpu_id=0,
                 anchor_assignment_method='iou',
                 joint_linking=False,
                 tracktor=None,
                 ):
        """

        Parameters
        ----------
        anchor_array
        use_motion
        match_top_k
        track_keep_alive_thresh
        new_track_iou_thresh
        track_nms_thresh
        gpu_id
        anchor_assignment
        joint_linking
        """
        self.use_motion = use_motion
        self.tracks = []
        self.all_track_id = 0
        self.pending_index = []
        self.conf_score_thresh = 0.1
        self.anchor_array = anchor_array
        self.next_frame_pred_index = []
        self.next_frame_pred_weights = []
        self.next_frame_pred_bbox = []
        self.waiting_update_tracks = []
        self.next_frame_ref_bbox = []
        self.last_frame = None
        self.k = match_top_k
        self.keep_alive = track_keep_alive_thresh
        self.new_track_iou_thresh = new_track_iou_thresh
        self.track_nms_thresh = track_nms_thresh
        self.frame_cache = None
        self.mx_ctx = mx.gpu(gpu_id)
        self.anchor_assignment_method = anchor_assignment_method
        self.joint_linking = joint_linking

        if motion_model == 'farneback':
            self.motion_estimator = FarneBeckFlowMotionEstimator()
        elif motion_model == 'no':
            self.motion_estimator = DummyMotionEstimator()
        else:
            raise ValueError("Unknown motion model: {}".format(motion_model))

    def process_frame_sequence(self, frame_iterator, tracktor):
        """
        Parameters
        ----------
        frame_iterator: each step it emits a tuple of (frame_id, frame_data)
        tracktor

        Returns
        -------
            results_iter:  a response iterator with one tuple (frame_id, frame_rst) per frame
        """
        for frame_id, frame in frame_iterator:
            logging.info('Processing Frame ID: {}'.format(frame_id))
            t_iter_start = time.time()

            # STEP 0: Prepare the tracktor with the new frame data
            motion_pred_data = tracktor.prepare_for_frame(frame)

            # STEP 1: Predict the new locations of the tracked bounding boxes in the tracker
            tracking_anchor_indices, tracking_anchor_weights, tracking_classes = self.motion_predict(frame, motion_pred_data)

            # STEP 2: Run the tracktor
            detection_bboxes, detection_anchor_indices, tracking_bboxes, extra_info \
                = tracktor.detect_and_track(frame,
                                            tracking_anchor_indices,
                                            tracking_anchor_weights,
                                            tracking_classes
                                            )
            if self.anchor_array is None:
                self.set_anchor_array(tracktor.anchors())

            # STEP 3: Update the tracker with detector responses
            self.update(detection_bboxes,
                        tracking_bboxes,
                        detection_anchor_indices,
                        tracking_anchor_indices,
                        tracking_anchor_weights,
                        tracking_classes,
                        extra_info)

            # yield the results of this frame
            results = self._produce_frame_result()

            elapsed = time.time() - t_iter_start
            logging.info("Total Tracking Runtime: {:2.4f} msec, {:.01f} FPS".format(
                elapsed * 1000, 1 / elapsed))
            yield frame_id, results

    @timeit
    def set_anchor_array(self, anchor_arracy):
        self.anchor_array = anchor_arracy
        self.anchor_tensor = mx.nd.array(self.anchor_array, ctx=self.mx_ctx, dtype=np.float32)

    @timeit
    def motion_predict(self, new_frame: np.ndarray, motion_pred_data):
        """
        Perform motion prediction and assign the predicted track locations to corresponding anchors for re-detection.
        It will update the following properties:
            next_frame_pred_index: indices of anchors that bear tracking information. Each track will be assigned to
                several anchors. They will vote in the re-detection processs.
            next_frame_pred_weights: weights in the re-detection voting
            next_frame_pred_bbox: motion-predicted locations of the tracked objects
            waiting_update_tracks: active tracks for re-detection
            next_frame_ref_bbox: original locations of the tracked objects

        Parameters
        ----------
        new_frame: BGR frame of this timestep
        motion_pred_data: extra data needed by the motion predictor
        Returns:
            next_frame_pred_index
            next_frame_pred_weights
        -------

        """
        # STEP 1: Find all active tracks
        active_track_boxes = []
        active_track_indices = []
        active_track_anchor_indices = []
        active_track_anchor_weights = []
        active_track_classes = []

        t_active = time.time()
        for track_idx, track in enumerate(self.tracks):
            if track.is_active():
                active_track_boxes.append(track.mean)
                active_track_indices.append(track_idx)
                src_idx, src_weights = track.source
                active_track_anchor_indices.append(src_idx)
                active_track_anchor_weights.append(src_weights)
                active_track_classes.append([track.class_id])
                logging.debug("active track {} with age: {}".format(track.track_id, track.age))
        active_track_boxes = np.array(active_track_boxes)
        active_track_anchor_indices = np.array(active_track_anchor_indices)
        active_track_anchor_weights = np.array(active_track_anchor_weights)
        e_active = time.time() - t_active
        logging.info('find active runtime: {:.05f}'.format(e_active))

        if len(active_track_boxes) > 0:
            # The following steps only happen if we have something to track

            # STEP 2: Warp the boxes according to flow
            predicted_track_boxes = self._motion_prediction(new_frame, active_track_boxes,
                                                            active_track_anchor_indices,
                                                            active_track_anchor_weights,
                                                            motion_pred_data,
                                                            skip=not self.use_motion)


            # STEP 3: Assign the warped boxes to anchor compositions
            tracking_anchor_indices, tracking_anchor_weights, tracking_anchor_validity = self._assign_box_to_anchors(
                predicted_track_boxes, method=self.anchor_assignment_method)

            # remove tracks becoming invalid after motion prediction
            invalid_track_numbers = np.nonzero(1 - tracking_anchor_validity)[0]
            logging.info("{}/{} tracks become invalid after motion prediction".format(len(invalid_track_numbers), len(active_track_boxes)))

            for i_invalid in invalid_track_numbers:
                self.tracks[active_track_indices[i_invalid]].mark_missed()

            # keep the valid tracks for re-detection
            valid_track_numbers = np.nonzero(tracking_anchor_validity)[0]
            self.next_frame_pred_index = tracking_anchor_indices[valid_track_numbers, ...]
            self.next_frame_pred_weights = tracking_anchor_weights[valid_track_numbers, ...]
            self.next_frame_pred_bbox = predicted_track_boxes[valid_track_numbers, ...]
            self.next_frame_pred_class = np.array(active_track_classes)[valid_track_numbers, ...]

            active_track_indices = np.array(active_track_indices)[valid_track_numbers, ...]
            active_track_boxes = active_track_boxes[valid_track_numbers, ...]

        else:
            # skip flow computation if there is no active track
            # just save the frame in cache
            predicted_track_boxes = self._motion_prediction(new_frame,
                                                            active_track_boxes,
                                                            active_track_anchor_indices, active_track_anchor_weights,
                                                            motion_pred_data, skip=True)

            assert len(predicted_track_boxes) == 0

            self.next_frame_pred_index = np.array([])
            self.next_frame_pred_weights = np.array([])
            self.next_frame_pred_bbox = np.array([])
            self.next_frame_pred_class = np.array([])

        self.waiting_update_tracks = active_track_indices
        self.next_frame_ref_bbox = active_track_boxes

        return self.next_frame_pred_index, self.next_frame_pred_weights, self.next_frame_pred_class

    @timeit
    def update(self, new_detections: np.ndarray, tracking_predictions: np.ndarray,
               detection_anchor_indices: np.ndarray,
               tracking_anchor_indices: np.ndarray, tracking_anchor_weights: np.ndarray,
               tracking_classes: np.ndarray,
               extra_info: dict = None):
        """
        Update the tracks according to tracking and detection predictions.
        Parameters
        ----------
        new_detections: Nx5 ndarray
        tracking_predictions: Mx5 ndarray
        extra_info: a dictionary with extra information

        Returns
        -------
        """
        # pylint: disable=too-many-nested-blocks
        t_pose_processing = time.time()

        logging.info("tracking predictions 's shape is {}".format(tracking_predictions.shape))
        logging.debug(tracking_predictions)
        logging.debug(self.waiting_update_tracks)

        detection_landmarks = extra_info['detection_landmarks'] if 'detection_landmarks' in extra_info else None
        tracking_landmarks = extra_info['tracking_landmarks'] if 'tracking_landmarks' in extra_info else None

        for t in self.tracks:
            t.predict()

        # STEP 1: track level NMS
        still_active_track_pred_indices = []
        still_active_track_indices = []

        if len(tracking_predictions) > 0:

            # class wise NMS
            keep_set = set()
            for c in set(tracking_classes.ravel().tolist()):
                class_pick = np.nonzero(tracking_classes == c)[0]
                keep_tracking_pred_nms_indices = nms_fallback(tracking_predictions[class_pick, ...], self.track_nms_thresh)
                for i_keep in keep_tracking_pred_nms_indices:
                    keep_set.add(class_pick[i_keep])

            still_active_track_pred_indices = []
            for i_pred, i_track in enumerate(self.waiting_update_tracks):
                if i_pred in keep_set:
                    self.tracks[i_track].update(tracking_predictions[i_pred, :],
                                                (tracking_anchor_indices[i_pred, :], tracking_anchor_weights[i_pred, :]),
                                                tracking_landmarks[i_pred, :] if tracking_landmarks is not None else None)
                else:
                    # suppressed tracks in the track NMS process will be marked as Missing
                    self.tracks[i_track].mark_missed()

                if self.tracks[i_track].is_active():
                    still_active_track_pred_indices.append(i_pred)
                    still_active_track_indices.append(i_track)

        # STEP 2: Remove New Detection Overlapping with Tracks
        if len(still_active_track_pred_indices) > 0 and len(new_detections) > 0:
            active_tracking_predictions = tracking_predictions[still_active_track_pred_indices, :]
            det_track_max_iou = bbox_iou(new_detections[:, :4], active_tracking_predictions[:, :4])
            same_class = new_detections[:, -1:] == (tracking_classes[still_active_track_pred_indices, :].T)
            # suppress all new detections that have high IOU with active tracks
            affinity = (det_track_max_iou * same_class).max(axis=1)
            keep_detection_indices = np.nonzero(affinity <= self.new_track_iou_thresh)[0]
        else:
            # otherwise simply keep all detections
            keep_detection_indices = list(range(len(new_detections)))
            active_tracking_predictions = np.array([])

        # STEP 3: New Track Initialization
        if len(keep_detection_indices) > 0:

            active_new_detections = new_detections[keep_detection_indices, :]
            # (Optional) STEP 3.a: Perform joint linking of body and head
            if self.joint_linking:
                tracking_classes = np.array(tracking_classes)
                body2face_link, face2body_link = \
                    self._link_face_body(active_new_detections,
                                         extra_info['detection_keypoints'][keep_detection_indices],
                                         active_tracking_predictions,
                                         extra_info['tracking_keypoints'][still_active_track_pred_indices],
                                         tracking_classes[still_active_track_pred_indices]
                                         )
            else:
                body2face_link, face2body_link = None, None

            new_tracks = []
            for idx, i_new_track in enumerate(keep_detection_indices):
                new_track = Track(new_detections[i_new_track, :4], self.all_track_id,
                                  (detection_anchor_indices[i_new_track, :], np.array([1])),
                                  keep_alive_thresh=self.keep_alive, class_id=new_detections[i_new_track, -1],
                                  attributes=detection_landmarks[i_new_track, :] if detection_landmarks is not None else None)
                if self.joint_linking:
                    if new_track.class_id == 0:
                        # new face track
                        if idx in face2body_link[0]:
                            logging.debug(idx, i_new_track, '0')
                            body_idx = face2body_link[0][idx]
                            if idx > body_idx:
                                new_track.link_to(new_tracks[body_idx])
                        elif idx in face2body_link[2]:
                            logging.debug(idx, i_new_track, '1')
                            body_idx = face2body_link[2][idx]
                            new_track.link_to(self.tracks[still_active_track_indices[body_idx]])

                    if new_track.class_id == 1:
                        # new body track
                        if idx in body2face_link[0]:
                            face_idx = body2face_link[0][idx]
                            if idx > face_idx:
                                new_track.link_to(new_tracks[face_idx])
                        elif idx in body2face_link[2]:
                            face_idx = body2face_link[2][idx]
                            new_track.link_to(self.tracks[still_active_track_indices[face_idx]])

                self.all_track_id += 1
                self.tracks.append(new_track)
                new_tracks.append(new_track)

        elapsed_post_processing = time.time() - t_pose_processing
        logging.info("total tracklets to now is {}, post-processing time: {:.05f} sec".format(
            self.all_track_id, elapsed_post_processing))

    @property
    def active_tracks(self):
        for t in self.tracks:
            if t.is_active():
                yield t

    def _motion_prediction(self, new_frame, tracked_boxes,
                           tracked_boxes_anchor_indices, tracked_boxes_anchor_weights,
                           motion_pred_data, skip=False):
        """
        Perform motion estimation of a set bounding boxes.
        Use either optical flow or SOT algorithms to predict the locations of these bounding boxes in the new frame
        Parameters
        ----------
        new_frame
        tracked_boxes
        tracked_boxes_anchor_indices
        tracked_boxes_anchor_weights
        skip

        Returns
        -------

        """
        if self.frame_cache is None:
            # this is the first frame
            self.frame_cache = self.motion_estimator.initialize(new_frame, motion_pred_data)
            predicted_bboxes = tracked_boxes
        else:
            # this is not the first frame
            predicted_bboxes, self.frame_cache = self.motion_estimator.predict_new_locations(
                self.frame_cache, tracked_boxes, new_frame, motion_pred_data,
                tracked_boxes_anchor_indices=tracked_boxes_anchor_indices,
                tracked_boxes_anchor_weights=tracked_boxes_anchor_weights,
                skip=skip)

        return predicted_bboxes

    def _assign_box_to_anchors(self, boxes: np.ndarray, method: str = 'avg', min_anchor_iou: float = 0.1):
        """
        The actual implementation of the assignment step.
        GPU acceleration is used because the number of anchors is huge
        Parameters
        ----------
        boxes: must have >1 boxes
        anchors

        Returns
        -------

        """
        t_start = time.time()

        t_iou = time.time()
        gpu_boxes = mx.nd.array(boxes, ctx=self.mx_ctx)
        anchor_track_iou = gpu_iou(self.anchor_tensor, gpu_boxes)
        elapsed_iou = time.time() - t_iou
        logging.info("iou computation runtime: {:.05f}".format(elapsed_iou))

        # get the top-k closest anchors instead of 1
        if method == 'max':
            tracking_anchor_ious, tracking_anchor_indices = mx.nd.topk(anchor_track_iou, axis=0, k=1,
                                                                       ret_typ='both', dtype='int32')
            tracking_anchor_ious = tracking_anchor_ious.T.asnumpy()
            tracking_anchor_indices = tracking_anchor_indices.T.asnumpy()
            tracking_anchor_weights = np.ones_like(tracking_anchor_indices)
        elif method == 'avg':
            tracking_anchor_ious, tracking_anchor_indices = mx.nd.topk(anchor_track_iou, axis=0, k=self.k,
                                                                       ret_typ='both', dtype='int32')
            tracking_anchor_ious = tracking_anchor_ious.T.asnumpy()
            tracking_anchor_indices = tracking_anchor_indices.T.asnumpy()
            tracking_anchor_weights = np.ones_like(tracking_anchor_indices) / self.k
        elif method == 'iou':
            t_sort = time.time()

            tracking_anchor_ious, tracking_anchor_indices = mx.nd.topk(anchor_track_iou, axis=0, k=self.k,
                                                                       ret_typ='both', dtype='int32')
            tracking_anchor_ious = tracking_anchor_ious.T.asnumpy()
            tracking_anchor_indices = tracking_anchor_indices.T.asnumpy()

            e_sort = time.time() - t_sort
            logging.info('sorting time: {:.05f}'.format(e_sort))
            tracking_anchor_weights = tracking_anchor_ious / tracking_anchor_ious.sum(axis=1)[..., None]
        else:
            raise ValueError("unknown anchor assignment method")

        max_track_anchor_ious = tracking_anchor_ious.max(axis=1)
        tracking_anchor_validity = max_track_anchor_ious >= min_anchor_iou

        elapsed_assign = time.time() - t_start
        logging.info("assigment runtime: {}".format(elapsed_assign))
        return tracking_anchor_indices, tracking_anchor_weights, tracking_anchor_validity

    @timeit
    def _link_face_body(self, detections, detection_kp, tracking_bbox, tracking_kp, tracking_classes,
                        thresh=1):
        det_cls_id = detections[:, -1]
        det_bbox = detections[:, :4]
        det_face_idx = (det_cls_id == 0).nonzero()[0]
        det_body_idx = (det_cls_id == 1).nonzero()[0]

        if len(tracking_bbox) > 0:
            track_cls_id = tracking_classes[:, 0]
            track_face_idx = (track_cls_id == 0).nonzero()[0]
            track_body_idx = (track_cls_id == 1).nonzero()[0]
            face_bboxes = np.concatenate((det_bbox[det_face_idx, :], tracking_bbox[track_face_idx, :4]), axis=0)
            head_kp = np.concatenate((detection_kp[det_body_idx, :], tracking_kp[track_body_idx, :]), axis=0)
        else:
            track_face_idx = []
            track_body_idx = []
            face_bboxes = det_bbox[det_face_idx, :]
            head_kp = detection_kp[det_body_idx, :]

        def get_cost(kp, bboxes):
            centers = (bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2
            wh = bboxes[:, [2, 3]] - bboxes[:, [0, 1]]

            rel_offset = (kp[:, :, None] - centers.T[None, :, :]) / wh.T[None, :, :]
            cost = np.abs(rel_offset).sum(axis=1)
            # pad the cost matrix by 1000
            max_size = max(cost.shape)
            out = np.ones((max_size, max_size)) * 1000
            out[:cost.shape[0], :cost.shape[1]] = cost
            return out

        matching_cost = get_cost(head_kp, face_bboxes)

        from ...utils.filesystem import try_import_munkres
        Munkres = try_import_munkres()
        m = Munkres()
        indexes = m.compute(matching_cost.tolist())
        d2d = dict()
        d2t = dict()
        t2t = dict()
        t2d = dict()

        logging.debug(len(det_body_idx), len(det_face_idx))
        logging.debug(head_kp, len(head_kp))
        logging.debug(face_bboxes, len(face_bboxes))

        # import pdb; pdb.set_trace()
        for row, column in indexes:
            if matching_cost[row][column] < thresh:
                if row < len(det_body_idx) and column < len(det_face_idx):
                    # detection-detection pair
                    d2d[det_body_idx[row]] = det_face_idx[column]
                elif row >= len(det_body_idx) and column >= len(det_face_idx):
                    # tracking-tracking pair
                    t2t[track_body_idx[row - len(det_body_idx)]] = track_face_idx[column - len(track_face_idx)]
                elif row < len(det_body_idx) and column >= len(det_face_idx):
                    # detection-tracking pair
                    d2t[det_body_idx[row]] = track_face_idx[column - len(track_face_idx)]
                elif row >= len(det_body_idx) and column < len(det_face_idx):
                    # tracking-detection pair
                    t2d[track_body_idx[row - len(det_body_idx)]] = det_face_idx[column]

        def reverse_link(link_dict):
            return {v:k for k, v in link_dict.items()}

        ret = [d2d, t2t, d2t, t2d]

        logging.debug(ret)
        return ret, [reverse_link(x) for x in [d2d, t2t, t2d, d2t]]

    @timeit
    def _produce_frame_result(self):
        tracked_objects = []
        for track in self.active_tracks:
            box = {
                'left': track.mean[0],
                'top': track.mean[1],
                'width': track.mean[2] - track.mean[0],
                'height': track.mean[3] - track.mean[1]
            }
            tid = track.display_id
            age = track.age
            classId = track.class_id
            obj = {
                'bbox': box,
                'track_id': tid,
                'age': age,
                'class_id':classId
            }
            if track.attributes is not None:
                obj['landmarks'] = track.attributes

            tracked_objects.append(obj)

        return tracked_objects
