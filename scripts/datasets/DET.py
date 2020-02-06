"""this script is used to prepare DET dataset for tracking,
which is Object detection in Large Scale Visual Recognition Challenge 2015 (ILSVRC2015)
Code adapted from https://github.com/STVIR/pysot"""
import argparse
import tarfile
from os.path import join, isdir, expanduser
from os import rename, system, chdir
import glob
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import json
import time
import sys
from concurrent import futures
import numpy as np
from gluoncv.utils.filesystem import try_import_cv2
from gluoncv.utils import download, makedirs

def parse_args():
    """DET dataset parameter."""
    parser = argparse.ArgumentParser(
        description='Download DET dataset and prepare for tracking')
    parser.add_argument('--download-dir', type=str, default='~/.mxnet/datasets/det/',
                        help='dataset directory on disk')
    parser.add_argument('--instance-size', type=int, default=511, help='instance image size')
    parser.add_argument('--num-threads', type=int, default=12, help='threads number')
    args = parser.parse_args()
    args.download_dir = expanduser(args.download_dir)
    return args

def download_det(args, overwrite=False):
    """download DET dataset and Unzip to download_dir"""
    _DOWNLOAD_URLS = [
    ('http://image-net.org/image/ILSVRC2015/ILSVRC2015_DET.tar.gz',
    'cbf602d89f2877fa8843392a1ffde03450a18d38'),
    ]
    if not isdir(args.download_dir):
        makedirs(args.download_dir)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=args.download_dir, overwrite=overwrite, sha1_hash=checksum)
        print(' dataset has already download completed')
        with tarfile.open(filename) as tar:
            tar.extractall(path=args.download_dir)
    if not isdir(join(args.download_dir, 'ILSVRC2015')):
        rename(join(args.download_dir, 'ILSVRC2015'), join(args.download_dir, 'ILSVRC'))


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
    cv2 = try_import_cv2()
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
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
    return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]


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
    target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
    target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
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


def crop_xml(args, xml, sub_set_crop_path, instance_size=511):
    """
    Dataset curation

    Parameters
    ----------
    xml: str , xml
    sub_set_crop_path: str, xml crop path
    instance_size: int, instance_size
    """
    cv2 = try_import_cv2()
    xmltree = ET.parse(xml)
    objects = xmltree.findall('object')

    frame_crop_base_path = join(sub_set_crop_path, xml.split('/')[-1].split('.')[0])
    if not isdir(frame_crop_base_path):
        makedirs(frame_crop_base_path)
    img_path = xml.replace('xml', 'JPEG').replace('Annotations', 'Data')
    im = cv2.imread(img_path)
    avg_chans = np.mean(im, axis=(0, 1))

    for id, object_iter in enumerate(objects):
        bndbox = object_iter.find('bndbox')
        bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
        z, x = crop_like_SiamFC(im, bbox, instance_size=instance_size, padding=avg_chans)
        cv2.imwrite(join(args.download_dir, frame_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(0, id)), z)
        cv2.imwrite(join(args.download_dir, frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, id)), x)

def par_crop(args):
    """
    Dataset curation,crop data and transform the format of a label
    """
    crop_path = join(args.download_dir, './crop{:d}'.format(args.instance_size))
    if not isdir(crop_path): makedirs(crop_path)
    VID_base_path = join(args.download_dir, './ILSVRC')
    ann_base_path = join(VID_base_path, 'Annotations/DET/train/')
    sub_sets = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i')
    for sub_set in sub_sets:
        sub_set_base_path = join(ann_base_path, sub_set)
        if 'a' == sub_set:
            xmls = sorted(glob.glob(join(sub_set_base_path, '*', '*.xml')))
        else:
            xmls = sorted(glob.glob(join(sub_set_base_path, '*.xml')))
        n_imgs = len(xmls)
        sub_set_crop_path = join(crop_path, sub_set)
        with futures.ProcessPoolExecutor(max_workers=args.num_threads) as executor:
            fs = [executor.submit(crop_xml, args, xml, sub_set_crop_path, args.instance_size) for xml in xmls]
            for i, f in enumerate(futures.as_completed(fs)):
                printProgress(i, n_imgs, prefix=sub_set, suffix='Done ', barLength=80)

def gen_json(args):
    """Format XML and transform json.
       generate train and val json, prepare for tracking dataloader"""
    js = {}
    VID_base_path = join(args.download_dir, './ILSVRC')
    ann_base_path = join(VID_base_path, 'Annotations/DET/train/')
    sub_sets = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i')
    for sub_set in sub_sets:
        sub_set_base_path = join(ann_base_path, sub_set)
        if 'a' == sub_set:
            xmls = sorted(glob.glob(join(sub_set_base_path, '*', '*.xml')))
        else:
            xmls = sorted(glob.glob(join(sub_set_base_path, '*.xml')))
        n_imgs = len(xmls)
        for f, xml in enumerate(xmls):
            print('subset: {} frame id: {:08d} / {:08d}'.format(sub_set, f, n_imgs))
            xmltree = ET.parse(xml)
            objects = xmltree.findall('object')

            video = join(sub_set, xml.split('/')[-1].split('.')[0])

            for id, object_iter in enumerate(objects):
                bndbox = object_iter.find('bndbox')
                bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                        int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                frame = '%06d' % (0)
                obj = '%02d' % (id)
                if video not in js:
                    js[video] = {}
                if obj not in js[video]:
                    js[video][obj] = {}
                js[video][obj][frame] = bbox

    train = {k:v for (k, v) in js.items() if 'i/' not in k}
    val = {k:v for (k, v) in js.items() if 'i/' in k}
    json.dump(train, open(join(args.download_dir, 'train.json'), 'w'),
              indent=4, sort_keys=True)
    json.dump(val, open(join(args.download_dir, 'val.json'), 'w'),
              indent=4, sort_keys=True)

def symlink(args):
    """Soft connection"""
    chdir(args.download_dir)
    system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2013_train ILSVRC/Annotations/DET/train/a')
    system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0000 ILSVRC/Annotations/DET/train/b')
    system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0001 ILSVRC/Annotations/DET/train/c')
    system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0002 ILSVRC/Annotations/DET/train/d')
    system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0003 ILSVRC/Annotations/DET/train/e')
    system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0004 ILSVRC/Annotations/DET/train/f')
    system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0005 ILSVRC/Annotations/DET/train/g')
    system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0006 ILSVRC/Annotations/DET/train/h')
    system('ln -sfb $PWD/ILSVRC/Annotations/DET/val ILSVRC/Annotations/DET/train/i')

    system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2013_train ILSVRC/Data/DET/train/a')
    system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0000 ILSVRC/Data/DET/train/b')
    system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0001 ILSVRC/Data/DET/train/c')
    system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0002 ILSVRC/Data/DET/train/d')
    system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0003 ILSVRC/Data/DET/train/e')
    system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0004 ILSVRC/Data/DET/train/f')
    system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0005 ILSVRC/Data/DET/train/g')
    system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0006 ILSVRC/Data/DET/train/h')
    system('ln -sfb $PWD/ILSVRC/Data/DET/val ILSVRC/Data/DET/train/i')

def main(args):
    # download DET dataset
    download_det(args)
    print('DET dataset has already download completed')
    symlink(args)
    # crop DET dataset for prepare for tracking
    par_crop(args)
    print('DET dataset has already crop completed')
    # generat DET json for prepare for tracking
    gen_json(args)
    print('DET dataset has already generat completed')

if __name__ == '__main__':
    since = time.time()
    args = parse_args()
    main(args)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

