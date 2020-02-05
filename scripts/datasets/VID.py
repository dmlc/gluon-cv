"""this script is used to prepare VID dataset for tracking,
which is Object detection from video in Large Scale Visual
Recognition Challenge 2015 (ILSVRC2015)"""
import json
from os.path import join, isdir, expanduser
from os import listdir, chdir, system
import glob
from concurrent import futures
import time
import argparse
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import tarfile
import sys
import numpy as np
from gluoncv.utils import download, makedirs
from gluoncv.utils.filesystem import try_import_cv2
cv2 = try_import_cv2()

def parse_args():
    """VID dataset parameter."""
    parser = argparse.ArgumentParser(
        description='Download VID dataset and prepare for tracking')
    parser.add_argument('--download-dir', type=str, default='~/.mxnet/datasets/vid/',
                        help='dataset directory on disk')
    parser.add_argument('--instance-size', type=int, default=511,
                        help='instance image size')
    parser.add_argument('--num-threads', type=int, default=12,
                        help='threads number')
    args = parser.parse_args()
    args.download_dir = expanduser(args.download_dir)
    return args

def download_VID(overwrite=False):
    """download VID dataset and Unzip to download_dir"""
    url = 'http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz'
    makedirs(args.download_dir)
    filename = download(url, path=args.download_dir, overwrite=overwrite)
    with tarfile.open(filename) as tar:
        tar.extractall(path=args.download_dir)

def parse_vid(ann_base_path, args):
    """
    Format XML and save it in JSON

    Parameters
    ----------
    ann_base_path: str, Annotations base path
    """
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
    json.dump(vid, open(join(args.download_dir, 'vid.json'), 'w'), indent=4, sort_keys=True)
    print('done!')

def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar

    Parameters
    ----------
        iteration: int : current iteration.
        total: int, total iterations.
        prefix: str, prefix string.
        suffix: str, suffix string.
        decimals: int, positive number of decimals in percent complete.
        barLength: int, character length of bar.
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percents = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()

def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    """
    crop image

    Parameters
    ----------
    image: np.array, image
    bbox: np or list, bbox coordinate [xmin,ymin,xmax,ymax]
    out_sz: int , crop image size

    Return:
        crop result
    """
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    """
    from center_x,center_y,s to get bbox

    Parameters
    ----------
    pos , x, bbox
    s , int, bbox size

    Return:
        [x_min,y_min,x_max,y_max]
    """
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instance_size=255, padding=(0, 0, 0)):
    """
    Dataset curation and avoid image resizing during training
    if the tight bounding box has size (w, h) and the context margin is p,
    then the scale factor s is chosen such that the area of the scaled rectangle is equal to a constant
    s(w+2p)×s(h+2p)=A.

    Parameters
    ----------
    image: np.array, image
    bbox: list or np.array, bbox
    context_amount: float, the amount of context to be half of the mean dimension
    exemplar_size: int, exemplar_size
    instance_size: int, instance_size

    Return:
        crop result exemplar z and instance x
    """
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instance_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instance_size, padding)
    return z, x


def crop_video(args, sub_set, video, crop_path, ann_base_path):
    """
    Dataset curation

    Parameters
    ----------
    sub_set: str , sub_set
    video: str, video number
    crop_path: str, crop_path
    ann_base_path: str, Annotations base path
    """
    video_crop_base_path = join(crop_path, sub_set, video)
    if not isdir(video_crop_base_path):
        makedirs(video_crop_base_path)
    sub_set_base_path = join(ann_base_path, sub_set)
    xmls = sorted(glob.glob(join(sub_set_base_path, video, '*.xml')))
    for xml in xmls:
        xmltree = ET.parse(xml)
        objects = xmltree.findall('object')
        objs = []
        filename = xmltree.findall('filename')[0].text
        # print(xml.replace('xml', 'JPEG').replace('Annotations', 'Data'))
        im = cv2.imread(xml.replace('xml', 'JPEG').replace('Annotations', 'Data'))
        avg_chans = np.mean(im, axis=(0, 1))
        for object_iter in objects:
            trackid = int(object_iter.find('trackid').text)
            bndbox = object_iter.find('bndbox')
            bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
            z, x = crop_like_SiamFC(im, bbox, instance_size=args.instance_size, padding=avg_chans)
            cv2.imwrite(join(args.download_dir, video_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(int(filename), trackid)), z)
            cv2.imwrite(join(args.download_dir, video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(filename), trackid)), x)

def par_crop(args, ann_base_path):
    """
    Dataset curation, crop data and transform the format of label

    Parameters
    ----------
    ann_base_path: str, Annotations base path
    """
    crop_path = join(args.download_dir, './crop{:d}'.format(int(args.instance_size)))
    if not isdir(crop_path):
        makedirs(crop_path)
    sub_sets = sorted({'a', 'b', 'c', 'd', 'e'})
    for sub_set in sub_sets:
        sub_set_base_path = join(ann_base_path, sub_set)
        videos = sorted(listdir(sub_set_base_path))
        n_videos = len(videos)
        with futures.ProcessPoolExecutor(max_workers=args.num_threads) as executor:
            fs = [executor.submit(crop_video, args, sub_set, video, crop_path, ann_base_path) for video in videos]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_videos, prefix=sub_set, suffix='Done ', barLength=40)

def gen_json(args):
    """transform json and generate train and val json, prepare for tracking dataloader"""
    print('load json (raw vid info), please wait 20 seconds~')
    vid = json.load(open(join(args.download_dir, 'vid.json'), 'r'))
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

    train = {k:v for (k, v) in snippets.items() if 'train' in k}
    val = {k:v for (k, v) in snippets.items() if 'val' in k}

    json.dump(train, open(os.path.join(args.download_dir, 'train.json'), 'w'),
              indent=4, sort_keys=True)
    json.dump(val, open(os.path.join(args.download_dir, 'val.json'), 'w'),
              indent=4, sort_keys=True)
    print('done!')

def symlink(args):
    """Soft connection"""
    chdir(args.download_dir)
    system('ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000 ILSVRC2015/Annotations/VID/train/a')
    system('ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0001 ILSVRC2015/Annotations/VID/train/b')
    system('ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0002 ILSVRC2015/Annotations/VID/train/c')
    system('ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0003 ILSVRC2015/Annotations/VID/train/d')
    system('ln -sfb $PWD/ILSVRC2015/Annotations/VID/val ILSVRC2015/Annotations/VID/train/e')

    system('ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000 ILSVRC2015/Data/VID/train/a')
    system('ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0001 ILSVRC2015/Data/VID/train/b')
    system('ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0002 ILSVRC2015/Data/VID/train/c')
    system('ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0003 ILSVRC2015/Data/VID/train/d')
    system('ln -sfb $PWD/ILSVRC2015/Data/VID/val ILSVRC2015/Data/VID/train/e')

def main(args):
    # download VID dataset
    download_VID(args)
    print('VID dataset has already download completed')
    VID_base_path = join(args.download_dir, 'ILSVRC2015')
    ann_base_path = join(VID_base_path, 'Annotations/VID/train/')
    symlink(args)
    # Format XML and save it in JSON
    parse_vid(ann_base_path, args)
    print('VID dataset json has already generat completed')
    # crop VID dataset for prepare for tracking
    par_crop(args, ann_base_path)
    print('VID dataset has already crop completed')
    # generat VID json for prepare for tracking
    gen_json(args)
    print('VID dataset has already generat completed')

if __name__ == '__main__':
    since = time.time()
    args = parse_args()
    main(args)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
