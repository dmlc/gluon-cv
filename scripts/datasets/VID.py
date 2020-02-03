import json
import xml.etree.ElementTree as ET
from os.path import join, isdir
from os import listdir, mkdir, makedirs
import glob
import xml.etree.ElementTree as ET
from concurrent import futures
import time
import os
import argparse
import numpy as np
from gluoncv.utils import download, makedirs
from gluoncv.utils.filesystem import try_import_cv2
import tarfile
import sys
cv2 = try_import_cv2()

def parse_args():
    """VID dataset parameter."""
    parser = argparse.ArgumentParser(
        description='Download VID dataset and prepare for tracking')
    parser.add_argument('--download-dir', type=str, default='~/.mxnet/datasets/VID/',
                        help='dataset directory on disk')
    parser.add_argument('--instanc-size', type=int, help='instanc image size')
    parser.add_argument('--num-threads', type=int, help='threads number')
    args = parser.parse_args()
    args.download_dir = os.path.expanduser(args.download_dir)
    return args

args = parse_args()

def download_VID(overwrite=False):
    url = 'http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz'
    makedirs(args.download_dir)
    filename = download(url, path=args.download_dir, overwrite=overwrite)
    with tarfile.open(filename) as tar:
        tar.extractall(path=args.download_dir)

def parse_vid(VID_base_path,ann_base_path,img_base_path,args):
    sub_sets = sorted({'a', 'b', 'c', 'd', 'e'})

    vid = []
    for sub_set in sub_sets:
        sub_set_base_path = join(ann_base_path, sub_set)
        videos = sorted(listdir(sub_set_base_path))
        s = []
        for vi, video in enumerate(videos):
            print('subset: {} video id: {:04d} / {:04d}'.format(sub_set, vi, len(videos)))
            v = dict()
            v['base_path'] = join(sub_set, video)
            v['frame'] = []
            video_base_path = join(sub_set_base_path, video)
            xmls = sorted(glob.glob(join(video_base_path, '*.xml')))
            for xml in xmls:
                f = dict()
                xmltree = ET.parse(xml)
                size = xmltree.findall('size')[0]
                frame_sz = [int(it.text) for it in size]
                objects = xmltree.findall('object')
                objs = []
                for object_iter in objects:
                    trackid = int(object_iter.find('trackid').text)
                    name = (object_iter.find('name')).text
                    bndbox = object_iter.find('bndbox')
                    occluded = int(object_iter.find('occluded').text)
                    o = dict()
                    o['c'] = name
                    o['bbox'] = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                                int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                    o['trackid'] = trackid
                    o['occ'] = occluded
                    objs.append(o)
                f['frame_sz'] = frame_sz
                f['img_path'] = xml.split('/')[-1].replace('xml', 'JPEG')
                f['objs'] = objs
                v['frame'].append(f)
            s.append(v)
        vid.append(s)
    print('save json (raw vid info), please wait 1 min~')
    json.dump(vid, open(join(args.download_dir,'vid.json'), 'w'), indent=4, sort_keys=True)
    print('done!')

# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x


def crop_video(args, sub_set, video, crop_path, instanc_size):
    video_crop_base_path = join(crop_path, sub_set, video)
    if not isdir(video_crop_base_path): makedirs(video_crop_base_path)

    sub_set_base_path = join(ann_base_path, sub_set)
    xmls = sorted(glob.glob(join(sub_set_base_path, video, '*.xml')))
    for xml in xmls:
        xmltree = ET.parse(xml)
        objects = xmltree.findall('object')
        objs = []
        filename = xmltree.findall('filename')[0].text

        im = cv2.imread(join(args.downloader_dir,xml.replace('xml', 'JPEG').replace('Annotations', 'Data')))
        avg_chans = np.mean(im, axis=(0, 1))
        for object_iter in objects:
            trackid = int(object_iter.find('trackid').text)
            bndbox = object_iter.find('bndbox')

            bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
            z, x = crop_like_SiamFC(im, bbox, instanc_size=instanc_size, padding=avg_chans)
            cv2.imwrite(join(args.download_dir,video_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(int(filename), trackid)), z)
            cv2.imwrite(join(args.download_dir,video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(filename), trackid)), x)


def par_crop(args,ann_base_path):
    crop_path = join(args.download_dir, './crop{:d}'.format(instanc_size))
    if not isdir(crop_path):
        mkdir(crop_path)
    for sub_set in sub_sets:
        sub_set_base_path = join(ann_base_path, sub_set)
        videos = sorted(listdir(sub_set_base_path))
        n_videos = len(videos)
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_video, args, sub_set, video, crop_path, instanc_size) for video in videos]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_videos, prefix=sub_set, suffix='Done ', barLength=40)


def check_size(frame_sz, bbox):
    min_ratio = 0.1
    max_ratio = 0.75
    # only accept objects >10% and <75% of the total frame
    area_ratio = np.sqrt((bbox[2]-bbox[0])*(bbox[3]-bbox[1])/float(np.prod(frame_sz)))
    ok = (area_ratio > min_ratio) and (area_ratio < max_ratio)
    return ok

def check_borders(frame_sz, bbox):
    dist_from_border = 0.05 * (bbox[2] - bbox[0] + bbox[3] - bbox[1])/2
    ok = (bbox[0] > dist_from_border) and (bbox[1] > dist_from_border) and \
         ((frame_sz[0] - bbox[2]) > dist_from_border) and \
         ((frame_sz[1] - bbox[3]) > dist_from_border)
    return ok

def gen_json(args):
    snippets = dict()
    n_snippets = 0
    n_videos = 0
    for subset in vid:
        for video in subset:
            n_videos += 1
            frames = video['frame']
            id_set = []
            id_frames = [[]] * 60
            for f, frame in enumerate(frames):
                objs = frame['objs']
                frame_sz = frame['frame_sz']
                for obj in objs:
                    trackid = obj['trackid']
                    occluded = obj['occ']
                    bbox = obj['bbox']
                    if trackid not in id_set:
                        id_set.append(trackid)
                        id_frames[trackid] = []
                    id_frames[trackid].append(f)
            if len(id_set) > 0:
                snippets[video['base_path']] = dict()
            for selected in id_set:
                frame_ids = sorted(id_frames[selected])
                sequences = np.split(frame_ids, np.array(np.where(np.diff(frame_ids) > 1)[0]) + 1)
                sequences = [s for s in sequences if len(s) > 1]
                for seq in sequences:
                    snippet = dict()
                    for frame_id in seq:
                        frame = frames[frame_id]
                        for obj in frame['objs']:
                            if obj['trackid'] == selected:
                                o = obj
                                continue
                        snippet[frame['img_path'].split('.')[0]] = o['bbox']
                    snippets[video['base_path']]['{:02d}'.format(selected)] = snippet
                    n_snippets += 1
            print('video: {:d} snippets_num: {:d}'.format(n_videos, n_snippets))
            
    train = {k:v for (k,v) in snippets.items() if 'train' in k}
    val = {k:v for (k,v) in snippets.items() if 'val' in k}

    json.dump(train, open(os.path.join(args.download_dir,'train.json'), 'w'),
              indent=4, sort_keys=True)
    json.dump(val, open(os.path.join(args.download_dir,'val.json'), 'w'),
              indent=4, sort_keys=True)
    print('done!')

def symlink(args):
    os.chdir(args.download_dir)
    os.system('ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000 ILSVRC2015/Annotations/VID/train/a')
    os.system('ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0001 ILSVRC2015/Annotations/VID/train/b')
    os.system('ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0002 ILSVRC2015/Annotations/VID/train/c')
    os.system('ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0003 ILSVRC2015/Annotations/VID/train/d')
    os.system('ln -sfb $PWD/ILSVRC2015/Annotations/VID/val ILSVRC2015/Annotations/VID/train/e')

    os.system('ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000 ILSVRC2015/Data/VID/train/a')
    os.system('ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0001 ILSVRC2015/Data/VID/train/b')
    os.system('ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0002 ILSVRC2015/Data/VID/train/c')
    os.system('ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0003 ILSVRC2015/Data/VID/train/d')
    os.system('ln -sfb $PWD/ILSVRC2015/Data/VID/val ILSVRC2015/Data/VID/train/e')

def main(args):
    # filename = download_VID()
    VID_base_path = os.path.join(args.download_dir,'ILSVRC2015')
    ann_base_path = join(VID_base_path, 'Annotations/VID/train/')
    img_base_path = join(VID_base_path, 'Data/VID/train/')
    symlink(args)
    parse_vid(VID_base_path, ann_base_path, img_base_path, args)
    par_crop(args, ann_base_path)
    gen_json(args)

if __name__ == '__main__':
    since = time.time()
    args = parse_args()
    main(args)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
