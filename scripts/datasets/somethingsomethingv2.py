"""This script is for preprocessing something-something-v2 dataset.
The code is largely borrowed from https://github.com/MIT-HAN-LAB/temporal-shift-module
and https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
"""

import os
import sys
import threading
import argparse

VIDEO_ROOT = '~/.mxnet/datasets/somethingsomethingv2/20bn-something-something-v2'
FRAME_ROOT = '~/.mxnet/datasets/somethingsomethingv2/20bn-something-something-v2-frames'

def parse_args():
    parser = argparse.ArgumentParser(description='prepare something-something-v2 dataset')
    parser.add_argument('--video_root', type=str, default='~/.mxnet/datasets/somethingsomethingv2/20bn-something-something-v2')
    parser.add_argument('--frame_root', type=str, default='~/.mxnet/datasets/somethingsomethingv2/20bn-something-something-v2-frames')
    parser.add_argument('--anno_root', type=str, default='~/.mxnet/datasets/somethingsomethingv2/annotations')
    parser.add_argument('--num_threads', type=int, default=100)
    parser.add_argument('--decode_video', action='store_true', default=False)
    parser.add_argument('--build_file_list', action='store_true', default=False)
    args = parser.parse_args()

    VIDEO_ROOT = os.path.expanduser(args.video_root)
    FRAME_ROOT = os.path.expanduser(args.frame_root)
    args.anno_root = os.path.expanduser(args.anno_root)
    return args

def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def extract(video, tmpl='%06d.jpg'):
    cmd = 'ffmpeg -i \"{}/{}\" -threads 1 -vf scale=-1:256 -q:v 0 \"{}/{}/%06d.jpg\"'.format(VIDEO_ROOT, video, FRAME_ROOT, video[:-5])
    os.system(cmd)

def target(video_list):
    for video in video_list:
        os.makedirs(os.path.join(FRAME_ROOT, video[:-5]))
        extract(video)

def decode_video(args):
    if not os.path.exists(VIDEO_ROOT):
        raise ValueError('Please download videos and set VIDEO_ROOT variable.')
    if not os.path.exists(FRAME_ROOT):
        os.makedirs(FRAME_ROOT)

    video_list = os.listdir(VIDEO_ROOT)
    splits = list(split(video_list, args.num_threads))

    threads = []
    for i, split in enumerate(splits):
        thread = threading.Thread(target=target, args=(split,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

def build_file_list(args):
    anno_root = args.anno_root
    if not os.path.exists(anno_root):
        raise ValueError('Please download annotations and set anno_root variable.')

    dataset_name = 'something-something-v2'
    with open(anno_root + '%s-labels.json' % dataset_name) as f:
        data = json.load(f)
    categories = []
    for i, (cat, idx) in enumerate(data.items()):
        assert i == int(idx)  # make sure the rank is right
        categories.append(cat)

    with open('category.txt', 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = [anno_root + '%s-validation.json' % dataset_name, anno_root + '%s-train.json' % dataset_name, anno_root + '%s-test.json' % dataset_name]
    files_output = ['val_videofolder.txt', 'train_videofolder.txt', 'test_videofolder.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            data = json.load(f)
        folders = []
        idx_categories = []
        for item in data:
            folders.append(item['id'])
            if 'test' not in filename_input:
                idx_categories.append(dict_categories[item['template'].replace('[', '').replace(']', '')])
            else:
                idx_categories.append(0)
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join(FRAME_ROOT, curFolder))
            if len(dir_files) == 0:
                print('video decoding fails at %s' (curFolder))
                sys.exit()
            output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))

if __name__ == '__main__':
    global args
    args = parse_args()

    if args.decode_video:
        print('Decoding videos to frames.')
        decode_video(args)

    if args.build_file_list:
        print('Generating training files.')
        build_file_list(args)
