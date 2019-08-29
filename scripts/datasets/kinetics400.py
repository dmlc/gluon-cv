"""This script is largely borrowed from https://github.com/open-mmlab/mmaction.
"""

import argparse
import sys
import os
import os.path as osp
import glob
import fnmatch
import random
import zipfile
from pipes import quote
from multiprocessing import Pool, current_process
import csv

def dump_frames(vid_item):

    from gluoncv.utils.filesystem import try_import_mmcv
    mmcv = try_import_mmcv()

    full_path, vid_path, vid_id = vid_item
    vid_name = vid_path.split('.')[0]
    out_full_path = osp.join(args.out_dir, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    vr = mmcv.VideoReader(full_path)
    for i in range(len(vr)):
        if vr[i] is not None:
            if args.new_width > 0 and args.new_height > 0:
                cur_img = mmcv.imresize(vr[i], (args.new_width, args.new_height))
            else:
                cur_img = vr[i]
            mmcv.imwrite(
                cur_img, '{}/img_{:05d}.jpg'.format(out_full_path, i + 1))
        else:
            print('[Warning] length inconsistent!'
                  'Early stop with {} out of {} frames'.format(i + 1, len(vr)))
            break
    print('{} done with {} frames'.format(vid_name, len(vr)))
    sys.stdout.flush()
    return True


def run_optical_flow(vid_item, dev_id=0):
    full_path, vid_path, vid_id = vid_item
    vid_name = vid_path.split('.')[0]
    out_full_path = osp.join(args.out_dir, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % args.num_gpu
    image_path = '{}/img'.format(out_full_path)
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    cmd = osp.join(args.df_path, 'build/extract_gpu') + \
        ' -f={} -x={} -y={} -i={} -b=20 -t=1 -d={} -s=1 -o={} -w={} -h={}' \
        .format(
        quote(full_path),
        quote(flow_x_path), quote(flow_y_path), quote(image_path),
        dev_id, args.out_format, args.new_width, args.new_height)

    os.system(cmd)
    print('{} {} done'.format(vid_id, vid_name))
    sys.stdout.flush()
    return True


def run_warp_optical_flow(vid_item, dev_id=0):
    full_path, vid_path, vid_id = vid_item
    vid_name = vid_path.split('.')[0]
    out_full_path = osp.join(args.out_dir, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % args.num_gpu
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    cmd = osp.join(args.df_path + 'build/extract_warp_gpu') + \
        ' -f={} -x={} -y={} -b=20 -t=1 -d={} -s=1 -o={}'.format(
            quote(full_path), quote(flow_x_path), quote(flow_y_path),
            dev_id, args.out_format)

    os.system(cmd)
    print('warp on {} {} done'.format(vid_id, vid_name))
    sys.stdout.flush()
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='prepare Kinetics400 dataset')
    parser.add_argument('--download_dir', type=str, default='~/.mxnet/datasets/kinetics400')
    parser.add_argument('--src_dir', type=str, default='~/.mxnet/datasets/kinetics400/train')
    parser.add_argument('--out_dir', type=str, default='~/.mxnet/datasets/kinetics400/rawframes_train')
    parser.add_argument('--frame_path', type=str, default='~/.mxnet/datasets/kinetics400/rawframes_train')
    parser.add_argument('--anno_dir', type=str, default='~/.mxnet/datasets/kinetics400/annotations')
    parser.add_argument('--out_list_path', type=str, default='~/.mxnet/datasets/kinetics400')
    parser.add_argument('--level', type=int, choices=[1, 2], default=2)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--flow_type', type=str, default=None, choices=[None, 'tvl1', 'warp_tvl1'])
    parser.add_argument('--df_path', type=str, default='./dense_flow', help='need dense flow implementation')
    parser.add_argument("--out_format", type=str, default='dir', choices=['dir', 'zip'], help='output format')
    parser.add_argument("--ext", type=str, default='mp4', choices=['avi', 'mp4'], help='video file extensions')
    parser.add_argument("--new_width", type=int, default=0, help='resize image width')
    parser.add_argument("--new_height", type=int, default=0, help='resize image height')
    parser.add_argument("--num_gpu", type=int, default=8, help='number of GPU')
    parser.add_argument("--resume", action='store_true', default=False, help='resume optical flow extraction instead of overwriting')
    parser.add_argument('--dataset', type=str, choices=['ucf101', 'kinetics400'], default='kinetics400')
    parser.add_argument('--rgb_prefix', type=str, default='img_')
    parser.add_argument('--flow_x_prefix', type=str, default='flow_x_')
    parser.add_argument('--flow_y_prefix', type=str, default='flow_y_')
    parser.add_argument('--num_split', type=int, default=1)
    parser.add_argument('--subset', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--format', type=str, default='rawframes', choices=['rawframes', 'videos'])
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--tiny_dataset', action='store_true', default=False)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--decode_video', action='store_true', default=False)
    parser.add_argument('--build_file_list', action='store_true', default=False)
    args = parser.parse_args()

    args.download_dir = os.path.expanduser(args.download_dir)
    args.src_dir = os.path.expanduser(args.src_dir)
    args.out_dir = os.path.expanduser(args.out_dir)
    args.frame_path = os.path.expanduser(args.frame_path)
    args.anno_dir = os.path.expanduser(args.anno_dir)
    args.out_list_path = os.path.expanduser(args.out_list_path)

    return args

def decode_video(args):

    if not osp.isdir(args.out_dir):
        print('Creating folder: {}'.format(args.out_dir))
        os.makedirs(args.out_dir)
    if args.level == 2:
        classes = os.listdir(args.src_dir)
        for classname in classes:
            new_dir = osp.join(args.out_dir, classname)
            if not osp.isdir(new_dir):
                print('Creating folder: {}'.format(new_dir))
                os.makedirs(new_dir)

    print('Reading videos from folder: ', args.src_dir)
    print('Extension of videos: ', args.ext)
    if args.level == 2:
        fullpath_list = glob.glob(args.src_dir + '/*/*.' + args.ext)
        done_fullpath_list = glob.glob(args.out_dir + '/*/*')
    elif args.level == 1:
        fullpath_list = glob.glob(args.src_dir + '/*.' + args.ext)
        done_fullpath_list = glob.glob(args.out_dir + '/*')
    print('Total number of videos found: ', len(fullpath_list))
    if args.resume:
        fullpath_list = set(fullpath_list).difference(set(done_fullpath_list))
        fullpath_list = list(fullpath_list)
        print('Resuming. number of videos to be done: ', len(fullpath_list))

    if args.level == 2:
        vid_list = list(map(lambda p: osp.join(
            '/'.join(p.split('/')[-2:])), fullpath_list))
    elif args.level == 1:
        vid_list = list(map(lambda p: p.split('/')[-1], fullpath_list))

    pool = Pool(args.num_worker)
    if args.flow_type == 'tvl1':
        pool.map(run_optical_flow, zip(
            fullpath_list, vid_list, range(len(vid_list))))
    elif args.flow_type == 'warp_tvl1':
        pool.map(run_warp_optical_flow, zip(
            fullpath_list, vid_list, range(len(vid_list))))
    else:
        pool.map(dump_frames, zip(
            fullpath_list, vid_list, range(len(vid_list))))

def parse_ucf101_splits(args):
    level = args.level
    class_ind = [x.strip().split()
                 for x in open(os.path.join(args.anno_dir, 'classInd.txt'))]
    class_mapping = {x[1]: int(x[0]) - 1 for x in class_ind}

    def line2rec(line):
        items = line.strip().split(' ')
        vid = items[0].split('.')[0]
        vid = '/'.join(vid.split('/')[-level:])
        label = class_mapping[items[0].split('/')[0]]
        return vid, label

    splits = []
    for i in range(1, 4):
        train_list = [line2rec(x) for x in open(
            os.path.join(args.anno_dir, 'trainlist{:02d}.txt'.format(i)))]
        test_list = [line2rec(x) for x in open(
            os.path.join(args.anno_dir, 'testlist{:02d}.txt'.format(i)))]
        splits.append((train_list, test_list))
    return splits

def parse_kinetics_splits(args):
    level = args.level
    csv_reader = csv.reader(
        open(os.path.join(args.anno_dir, 'kinetics_train.csv')))
    # skip the first line
    next(csv_reader)

    def convert_label(s):
        return s.replace('"', '').replace(' ', '_')

    labels_sorted = sorted(
        set([convert_label(row[0]) for row in csv_reader]))
    class_mapping = {label: i for i, label in enumerate(labels_sorted)}

    def list2rec(x, test=False):
        if test:
            vid = '{}_{:06d}_{:06d}'.format(x[0], int(x[1]), int(x[2]))
            label = -1  # label unknown
            return vid, label
        else:
            vid = '{}_{:06d}_{:06d}'.format(x[1], int(x[2]), int(x[3]))
            if level == 2:
                vid = '{}/{}'.format(convert_label(x[0]), vid)
            else:
                assert level == 1
            label = class_mapping[convert_label(x[0])]
            return vid, label

    csv_reader = csv.reader(
        open(os.path.join(args.anno_dir, 'kinetics_train.csv')))
    next(csv_reader)
    train_list = [list2rec(x) for x in csv_reader]
    csv_reader = csv.reader(
        open(os.path.join(args.anno_dir, 'kinetics_val.csv')))
    next(csv_reader)
    val_list = [list2rec(x) for x in csv_reader]
    csv_reader = csv.reader(
        open(os.path.join(args.anno_dir, 'kinetics_test.csv')))
    next(csv_reader)
    test_list = [list2rec(x, test=True) for x in csv_reader]

    return ((train_list, val_list, test_list), )

def parse_directory(path, key_func=lambda x: x[-11:],
                    rgb_prefix='img_',
                    flow_x_prefix='flow_x_',
                    flow_y_prefix='flow_y_',
                    level=1):
    """
    Parse directories holding extracted frames from standard benchmarks
    """
    print('parse frames under folder {}'.format(path))
    if level == 1:
        frame_folders = glob.glob(os.path.join(path, '*'))
    elif level == 2:
        frame_folders = glob.glob(os.path.join(path, '*', '*'))
    else:
        raise ValueError('level can be only 1 or 2')

    def count_files(directory, prefix_list):
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x+'*')) for x in prefix_list]
        return cnt_list

    # check RGB
    frame_dict = {}
    for i, f in enumerate(frame_folders):
        all_cnt = count_files(f, (rgb_prefix, flow_x_prefix, flow_y_prefix))
        k = key_func(f)

        x_cnt = all_cnt[1]
        y_cnt = all_cnt[2]
        if x_cnt != y_cnt:
            raise ValueError(
                'x and y direction have different number '
                'of flow images. video: ' + f)
        if i % 200 == 0:
            print('{} videos parsed'.format(i))

        frame_dict[k] = (f, all_cnt[0], x_cnt)

    print('frame folder analysis done')
    return frame_dict

def build_split_list(split, frame_info, shuffle=False):

    def build_set_list(set_list):
        rgb_list, flow_list = list(), list()
        for item in set_list:
            if item[0] not in frame_info:
                # print("item:", item)
                continue
            elif frame_info[item[0]][1] > 0:
                rgb_cnt = frame_info[item[0]][1]
                flow_cnt = frame_info[item[0]][2]
                rgb_list.append('{} {} {}\n'.format(
                    item[0], rgb_cnt, item[1]))
                flow_list.append('{} {} {}\n'.format(
                    item[0], flow_cnt, item[1]))
            else:
                rgb_list.append('{} {}\n'.format(
                    item[0], item[1]))
                flow_list.append('{} {}\n'.format(
                    item[0], item[1]))
        if shuffle:
            random.shuffle(rgb_list)
            random.shuffle(flow_list)
        return rgb_list, flow_list

    train_rgb_list, train_flow_list = build_set_list(split[0])
    test_rgb_list, test_flow_list = build_set_list(split[1])
    return (train_rgb_list, test_rgb_list), (train_flow_list, test_flow_list)

def build_file_list(args):

    if args.level == 2:
        def key_func(x): return '/'.join(x.split('/')[-2:])
    else:
        def key_func(x): return x.split('/')[-1]

    if args.format == 'rawframes':
        frame_info = parse_directory(args.frame_path,
                                     key_func=key_func,
                                     rgb_prefix=args.rgb_prefix,
                                     flow_x_prefix=args.flow_x_prefix,
                                     flow_y_prefix=args.flow_y_prefix,
                                     level=args.level)
    else:
        if args.level == 1:
            video_list = glob.glob(osp.join(args.frame_path, '*'))
        elif args.level == 2:
            video_list = glob.glob(osp.join(args.frame_path, '*', '*'))
        frame_info = {osp.relpath(
            x.split('.')[0], args.frame_path): (x, -1, -1) for x in video_list}

    if args.dataset == 'ucf101':
        split_tp = parse_ucf101_splits(args)
    elif args.dataset == 'kinetics400':
        split_tp = parse_kinetics_splits(args)
    assert len(split_tp) == args.num_split

    out_path = args.out_list_path
    if len(split_tp) > 1:
        for i, split in enumerate(split_tp):
            lists = build_split_list(split_tp[i], frame_info,
                                     shuffle=args.shuffle)
            filename = '{}_train_split_{}_{}.txt'.format(args.dataset,
                                                         i + 1, args.format)
            with open(osp.join(out_path, filename), 'w') as f:
                f.writelines(lists[0][0])
            filename = '{}_val_split_{}_{}.txt'.format(args.dataset,
                                                       i + 1, args.format)
            with open(osp.join(out_path, filename), 'w') as f:
                f.writelines(lists[0][1])
    else:
        lists = build_split_list(split_tp[0], frame_info,
                                 shuffle=args.shuffle)
        filename = '{}_{}_list_{}.txt'.format(args.dataset,
                                              args.subset,
                                              args.format)
        if args.subset == 'train':
            ind = 0
        elif args.subset == 'val':
            ind = 1
        elif args.subset == 'test':
            ind = 2
        with open(osp.join(out_path, filename), 'w') as f:
            f.writelines(lists[0][ind])

def download_kinetics400(args):

    print('Start downloading Kinetics400 annotation files.')
    download_kinetics400_anno(args)
    print('Download complete.')

    download_kinetics400_videos(args)

def download_kinetics400_anno(args):

    target_dir = args.download_dir
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    train_file = 'https://deepmind.com/documents/66/kinetics_train.zip'
    val_file = 'https://deepmind.com/documents/65/kinetics_val.zip'
    test_file = 'wget https://deepmind.com/documents/81/kinetics_test.zip'

    os.system('wget -P %s %s' % (target_dir, train_file))
    os.system('wget -P %s %s' % (target_dir, val_file))
    os.system('wget -P %s %s' % (target_dir, test_file))

    with zipfile.ZipFile(os.path.join(target_dir, 'kinetics_train.zip')) as zf:
        zf.extractall(path=target_dir)

    with zipfile.ZipFile(os.path.join(target_dir, 'kinetics_val.zip')) as zf:
        zf.extractall(path=target_dir)

    with zipfile.ZipFile(os.path.join(target_dir, 'kinetics_test.zip')) as zf:
        zf.extractall(path=target_dir)

def download_kinetics400_videos(args):

    print('Please refer to the official Kinetics crawler for downloading the videos \
        at https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics. ')

if __name__ == '__main__':
    global args
    args = parse_args()

    if args.download:
        print('Downloading Kinetics400 dataset.')
        download_kinetics400(args)

    if args.decode_video:
        print('Decoding videos to frames.')
        decode_video(args)

    if args.build_file_list:
        print('Generating training files.')
        build_file_list(args)
