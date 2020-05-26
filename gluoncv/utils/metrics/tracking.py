""" SiamRPN metrics """
import numpy as np
from ..filesystem import try_import_colorama

def Iou(rect1, rect2):
    """
    caculate interection over union

    Parameters
    ----------
        rect1: list or np.array, rectangle1
        rect2: list or np.array, rectangle2

    Returns
    -------
    iou
    """
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2-x1) * (y2-y1)
    target_a = (tx2-tx1) * (ty2 - ty1)
    inter = ww * hh
    iou = inter / (area + target_a - inter)
    return iou

def overlap_ratio(rect1, rect2):
    """Compute overlap ratio between two rects

    Parameters
    ----------
        rect1 : nd.array
            2d array of N x [x,y,w,h]
        rect2 : nd.array
            2d array of N x [x,y,w,h]

    Return
    ----------
        IOU
    """
    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0]+rect1[:, 2], rect2[:, 0]+rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1]+rect1[:, 3], rect2[:, 1]+rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2]*rect1[:, 3] + rect2[:, 2]*rect2[:, 3] - intersect
    iou = intersect / union
    iou = np.maximum(np.minimum(1, iou), 0)
    return iou

def success_overlap(gt_bb, result_bb, n_frame):
    """get success_overlap score

    Parameters
    ----------
        result_bb : nd.array
            2d array of N x [x,y,w,h]
        n_frame : int
            frame number

    Return
    ----------
        success score
    """
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success = np.zeros(len(thresholds_overlap))
    iou = np.ones(len(gt_bb)) * (-1)
    mask = np.sum(gt_bb[:, 2:] > 0, axis=1) == 2
    iou[mask] = overlap_ratio(gt_bb[mask], result_bb[mask])
    for i, per_threshold in enumerate(thresholds_overlap):
        success[i] = np.sum(iou > per_threshold) / float(n_frame)
    return success

def success_error(gt_center, result_center, thresholds, n_frame):
    """get success_error score

    Parameters
    ----------
        gt_center : np.ndarray
            2d array of N x [x,y,w,h]
        result_center : np.ndarray
            2d array of N x [x,y,w,h]
        thresholds : float
            error float
        n_frame : int
            frame number

    Return
    ----------
        success_error score
    """
    success = np.zeros(len(thresholds))
    dist = np.ones(len(gt_center)) * (-1)
    mask = np.sum(gt_center > 0, axis=1) == 2
    dist[mask] = np.sqrt(np.sum(
        np.power(gt_center[mask] - result_center[mask], 2), axis=1))
    for i, per_threshold in enumerate(thresholds):
        success[i] = np.sum(dist <= per_threshold) / float(n_frame)
    return success


class OPEBenchmark:
    """
    SiamRPN OPEBenchmark have eval_success, precision to select.
    eval_success is  distance between the center point of the predicted position
    precision is Compute overlap ratio between two rects through thresholds_overlap
    Parameters
    ----------
        dataset
            dataset Benchmark
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def convert_bb_to_center(self, bboxes):
        return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                         (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T

    def convert_bb_to_norm_center(self, bboxes, gt_wh):
        return self.convert_bb_to_center(bboxes) / (gt_wh+1e-16)

    def eval_success(self, eval_trackers=None):
        """eval_success is  distance between the center point of the predicted position
           and the center position marked in the benchmark
        Parameters
        ----------
            eval_trackers: list of tracker name or single tracker name
        Return
        ----------
            return: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        success_ret = {}
        for tracker_name in eval_trackers:
            success_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                                                      tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                success_ret_[video.name] = success_overlap(gt_traj, tracker_traj, n_frame)
            success_ret[tracker_name] = success_ret_
        return success_ret

    def eval_precision(self, eval_trackers=None):
        """
        eval model precision in eval_precision
        Parameters
        ----------
            eval_trackers: list of tracker name or single tracker name
        Return
        ----------
            return: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        precision_ret = {}
        for tracker_name in eval_trackers:
            precision_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                                                      tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                gt_center = self.convert_bb_to_center(gt_traj)
                tracker_center = self.convert_bb_to_center(tracker_traj)
                thresholds = np.arange(0, 51, 1)
                precision_ret_[video.name] = success_error(gt_center, tracker_center,
                                                           thresholds, n_frame)
            precision_ret[tracker_name] = precision_ret_
        return precision_ret

    def eval_norm_precision(self, eval_trackers=None):
        """
        eval model precision in eval_norm_precision
        Parameters
        ----------
            eval_trackers: list of tracker name or single tracker name
        Return
        ----------
            return: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        norm_precision_ret = {}
        for tracker_name in eval_trackers:
            norm_precision_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                                                      tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                gt_center_norm = self.convert_bb_to_norm_center(gt_traj, gt_traj[:, 2:4])
                tracker_center_norm = self.convert_bb_to_norm_center(tracker_traj, gt_traj[:, 2:4])
                thresholds = np.arange(0, 51, 1) / 100
                norm_precision_ret_[video.name] = success_error(gt_center_norm,
                                                                tracker_center_norm,
                                                                thresholds,
                                                                n_frame)
            norm_precision_ret[tracker_name] = norm_precision_ret_
        return norm_precision_ret

    def show_result(self, success_ret, precision_ret=None,
                    norm_precision_ret=None, show_video_level=False, helight_threshold=0.6):
        """pretty print result
        Parameters
        ----------
            success_ret: returned dict from function eval
        """
        colorama = try_import_colorama()
        tracker_auc = {}
        for tracker_name in success_ret.keys():
            auc = np.mean(list(success_ret[tracker_name].values()))
            tracker_auc[tracker_name] = auc
        tracker_auc_ = sorted(tracker_auc.items(),
                              key=lambda x: x[1],
                              reverse=True)[:20]
        tracker_names = [x[0] for x in tracker_auc_]


        tracker_name_len = max((max([len(x) for x in success_ret.keys()])+2), 12)
        header = ("|{:^"+str(tracker_name_len)+"}|{:^9}|{:^16}|{:^11}|").format("Tracker name",
                                                                                "Success",
                                                                                "Norm Precision",
                                                                                "Precision")
        formatter = "|{:^"+str(tracker_name_len)+"}|{:^9.3f}|{:^16.3f}|{:^11.3f}|"
        print('-'*len(header))
        print(header)
        print('-'*len(header))
        for tracker_name in tracker_names:
            success = tracker_auc[tracker_name]
            if precision_ret is not None:
                precision = np.mean(list(precision_ret[tracker_name].values()), axis=0)[20]
            else:
                precision = 0
            if norm_precision_ret is not None:
                norm_precision = np.mean(list(norm_precision_ret[tracker_name].values()),
                                         axis=0)[20]
            else:
                norm_precision = 0
            print(formatter.format(tracker_name, success, norm_precision, precision))
        print('-'*len(header))

        if show_video_level and len(success_ret) < 10 \
                and precision_ret is not None \
                and len(precision_ret) < 10:
            print("\n\n")
            header1 = "|{:^21}|".format("Tracker name")
            header2 = "|{:^21}|".format("Video name")
            for tracker_name in success_ret.keys():
                header1 += ("{:^21}|").format(tracker_name)
                header2 += "{:^9}|{:^11}|".format("success", "precision")
            print('-'*len(header1))
            print(header1)
            print('-'*len(header1))
            print(header2)
            print('-'*len(header1))
            videos = list(success_ret[tracker_name].keys())
            for video in videos:
                row = "|{:^21}|".format(video)
                for tracker_name in success_ret.keys():
                    success = np.mean(success_ret[tracker_name][video])
                    precision = np.mean(precision_ret[tracker_name][video])
                    success_str = "{:^9.3f}".format(success)
                    if success < helight_threshold:
                        row += f'{colorama.Fore.RED}{success_str}{colorama.Style.RESET_ALL}|'
                    else:
                        row += success_str+'|'
                    precision_str = "{:^11.3f}".format(precision)
                    if precision < helight_threshold:
                        row += f'{colorama.Fore.RED}{precision_str}{colorama.Style.RESET_ALL}|'
                    else:
                        row += precision_str+'|'
                print(row)
            print('-'*len(header1))
