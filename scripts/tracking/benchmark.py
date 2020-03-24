""" SiamRPN benchmark
Code adapted from https://github.com/STVIR/pysot"""
from glob import glob
import argparse
import os
from multiprocessing import Pool
from tqdm import tqdm
from gluoncv.utils.metrics.tracking import OPEBenchmark
from gluoncv.data.otb.tracking import OTBTracking as OTBDataset

def parse_args():
    """ benchmark test."""
    parser = argparse.ArgumentParser(description='tracking evaluation')
    parser.add_argument('--tracker-path', '-p', type=str, help='test result path')
    parser.add_argument('--dataset', '-d', default='OTB2015', type=str, help='dataset name')
    parser.add_argument('--num', '-n', default=1, type=int, help='number of thread to eval')
    parser.add_argument('--tracker-prefix', '-t', type=str, help='tracker name')
    parser.add_argument('--show-video-level', '-s', action='store_true')
    parser.add_argument('--test-dataset', type=str, help='test_json dataset dir')
    parser.set_defaults(show_video_level=False)
    opt = parser.parse_args()
    return opt

def main():
    """SiamRPN benchmark.
    evaluation according to txt of test result.now supports benchmark is Success and Precision
    Currently only supports test OTB 2015 dataset.

    Parameters
    ----------
    tracker_path : str, txt of test result path.
    tracker_prefix : str, model name.
    test_dataset : str, Path to test label json.
    """
    opt = parse_args()
    tracker_dir = os.path.join(opt.tracker_path, opt.dataset)
    trackers = glob(os.path.join(opt.tracker_path,
                                 opt.dataset,
                                 opt.tracker_prefix+'*'))
    trackers = [x.split('/')[-1] for x in trackers]
    assert len(trackers) > 0
    opt.num = min(opt.num, len(trackers))
    dataset = OTBDataset(name=opt.dataset, dataset_root=opt.test_dataset, load_img=False)
    dataset.set_tracker(tracker_dir, trackers)
    benchmark = OPEBenchmark(dataset)
    success_ret = {}
    with Pool(processes=opt.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_success, trackers),
                        desc='eval success', total=len(trackers), ncols=100):
            success_ret.update(ret)
    precision_ret = {}
    with Pool(processes=opt.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_precision, trackers),
                        desc='eval precision', total=len(trackers), ncols=100):
            precision_ret.update(ret)
    benchmark.show_result(success_ret, precision_ret,
                          show_video_level=opt.show_video_level)


if __name__ == '__main__':
    main()
