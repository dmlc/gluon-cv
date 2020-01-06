from gluoncv.utils.metrics.tracking import OPEBenchmark
from glob import glob
import argparse
import os
from gluoncv.data.otb.tracking import OTBTracking as OTBDataset
from multiprocessing import Pool
from tqdm import tqdm

parser = argparse.ArgumentParser(description='tracking evaluation')
parser.add_argument('--tracker_path', '-p',type=str,help='tracker result path')
parser.add_argument('--dataset', '-d', default='OTB2015',type=str,help='dataset name')
parser.add_argument('--num', '-n', default=1, type=int,help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t',type=str, help='tracker name')
parser.add_argument('--show_video_level', '-s', dest='show_video_level',action='store_true')
parser.add_argument('--test_dataset',type=str,help='test_json dataset dir')
parser.set_defaults(show_video_level=False)
args = parser.parse_args()

def benchmark():
    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob(os.path.join(args.tracker_path,
                                 args.dataset,
                                 args.tracker_prefix+'*'))
    trackers = [x.split('/')[-1] for x in trackers]
    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))
    root = os.path.join(args.test_dataset, args.dataset)
    dataset = OTBDataset(args.dataset, root)
    dataset.set_tracker(tracker_dir, trackers)
    benchmark = OPEBenchmark(dataset)
    success_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
            trackers), desc='eval success', total=len(trackers), ncols=100):
            success_ret.update(ret)
    precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
            trackers), desc='eval precision', total=len(trackers), ncols=100):
            precision_ret.update(ret)
    benchmark.show_result(success_ret, precision_ret,
            show_video_level=args.show_video_level)

if __name__ == '__main__':
    benchmark()