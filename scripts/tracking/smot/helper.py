import motmetrics as mm
import numpy as np
import os
import sys


def get_gt_pred_pairs(gt_dir, pred_dir, gt_prefix='gt_', pred_prefix='pred_', iou_thresh=0.5):
    gt_files = os.listdir(gt_dir)
    pred_files = os.listdir(pred_dir)

    gt_name_dict = {x[len(gt_prefix):]: x for x in gt_files if x.startswith(gt_prefix)}
    pred_name_dict = {x[len(pred_prefix):]: x for x in pred_files if x.startswith(pred_prefix)}

    pairs = [(x, os.path.join(gt_dir, gt_name_dict[x]), os.path.join(pred_dir, pred_name_dict[x]), iou_thresh)
             for x in gt_name_dict if x in pred_name_dict]
    return pairs


def get_sum(name, result_dict):
    return sum(x[name][0] for x in result_dict.values() if x['num_frames'][0] != 0)


def get_simple_metrics(name):

    def func(result_dict):
        return get_sum(name, result_dict)

    return func


num_transfer = get_simple_metrics('num_transfer')
num_objects = get_simple_metrics('num_objects')
num_misses = get_simple_metrics('num_misses')
num_fp = get_simple_metrics('num_false_positives')
num_sw = get_simple_metrics('num_switches')
num_pred = get_simple_metrics('num_predictions')
num_frag = get_simple_metrics("num_fragmentations")
idtp = get_simple_metrics('idtp')
num_det = get_simple_metrics('num_detections')
mt = get_simple_metrics('mostly_tracked')
pt = get_simple_metrics('partially_tracked')
ml = get_simple_metrics('mostly_lost')


def mota(result_dict):
    num = num_misses(result_dict) + num_sw(result_dict) + num_fp(result_dict)
    denum = num_objects(result_dict)
    return 1 - num / denum


def idf1(result_dict):
    num = idtp(result_dict)
    denum = num_objects(result_dict) + num_pred(result_dict)
    return 2 * num / denum


def idr(result_dict):
    num = idtp(result_dict)
    return num / num_objects(result_dict)


def idp(result_dict):
    num = idtp(result_dict)
    return num / num_pred(result_dict)


def precision(result_dict):
    return num_det(result_dict) / num_pred(result_dict)


def recall(result_dict):
    return num_det(result_dict) / num_objects(result_dict)


def f1(result_dict):
    num = num_det(result_dict)
    denum = num_objects(result_dict) + num_pred(result_dict)
    return 2 * num / denum


def num_tracks(result_dict):
    return mt(result_dict) + pt(result_dict) + ml(result_dict)


def array2dict(data_array):
    data_dict = {}
    for item in data_array:
        fid = int(item[0])
        data_dict.setdefault(fid, list())
        data_dict[fid].append(item)
    return data_dict


def update_acc(acc, oframe_data, hframe_data, min_iou=0.5):
    ob = np.array([x[2:6] for x in oframe_data])
    hb = np.array([x[2:6] for x in hframe_data])
    dist = mm.distances.iou_matrix(ob, hb, max_iou=1 - min_iou)
    oid = [x[1] for x in oframe_data]
    hid = [x[1] for x in hframe_data]
    acc.update(oid, hid, dist)
    return acc


def run_video(video_name, gt_name, pred_name, iou=0.5):
    gdata = np.load(gt_name, allow_pickle=True)
    pdata = np.load(pred_name, allow_pickle=True)
    gdict = array2dict(gdata)
    pdict = array2dict(pdata)

    acc = mm.MOTAccumulator(auto_id=True)

    for i in range(len(pdict)):

        try:
            acc = update_acc(acc, gdict[i] if i in gdict else [], pdict[i] if i in pdict else [], min_iou=iou)
        except:
            print("failed on video {}".format(video_name))
            sys.stdout.flush()
            break

    mh = mm.metrics.create()
    video_results = mh.compute(acc,
                               metrics=['num_frames', 'mota', 'motp', 'num_transfer', 'idf1', 'num_switches', 'mostly_tracked',
                                        'partially_tracked', 'mostly_lost', 'precision',
                                        'recall', 'idtp', 'num_objects', 'num_predictions',
                                        'num_misses', 'num_false_positives', 'num_detections'], name='acc')
    return video_name, video_results
