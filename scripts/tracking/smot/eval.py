import argparse
import os
import sys
from multiprocessing import Pool
from terminaltables import AsciiTable

from helper import *


parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str, default='smot')
parser.add_argument('--gt-dir', type=str, required=True)
parser.add_argument('--pred-dir', type=str, required=True)
parser.add_argument('--min-iou', type=float, default=0.5)
parser.add_argument('--num-worker', type=int, default=32)


if __name__ == '__main__':
    args = parser.parse_args()

    gt_pred_pairs = get_gt_pred_pairs(args.gt_dir, args.pred_dir, iou_thresh=args.min_iou)

    pool = Pool(args.num_worker)
    results = dict(pool.starmap(run_video, gt_pred_pairs))
    pool.close()
    pool.join()
    print("run on {} videos".format(len(gt_pred_pairs)))

    headers = ['MOTA', 'IDF1', 'IDR', 'IDP', 'N. Trans', 'FP','FN', 'IDsw.', 'MT/GT', 'Prec.', 'Rec.', 'F1']
    data = [mota(results), idf1(results), idr(results), idp(results), num_transfer(results), num_fp(results), num_misses(results), num_sw(results),
            '{}/{}'.format(mt(results), num_tracks(results)), precision(results), recall(results), f1(results)]

    table = AsciiTable([headers, data], title='Tracking Results: {}'.format(args.model_name))
    print(table.table)

