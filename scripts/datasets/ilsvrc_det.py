"""this script is used to prepare DET dataset for tracking,
which is Object detection in Large Scale Visual Recognition Challenge 2015 (ILSVRC2015)
Code adapted from https://github.com/STVIR/pysot"""
import argparse
import tarfile
import os
import glob
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import json
import time
from concurrent import futures
import numpy as np
from gluoncv.utils.filesystem import try_import_cv2
from gluoncv.utils import download, makedirs
from gluoncv.utils.data.tracking import crop_like_SiamFC, printProgress

def parse_args():
    """DET dataset parameter."""
    parser = argparse.ArgumentParser(
        description='Download DET dataset and prepare for tracking')
    parser.add_argument('--download-dir', type=str, default='~/.mxnet/datasets/tracking/det/',
                        help='dataset directory on disk')
    parser.add_argument('--instance-size', type=int, default=511, help='instance image size')
    parser.add_argument('--num-threads', type=int, default=12, help='threads number')
    args = parser.parse_args()
    args.download_dir = os.path.expanduser(args.download_dir)
    return args

def download_det(args, overwrite=False):
    """download DET dataset and Unzip to download_dir"""
    _DOWNLOAD_URLS = [
    ('http://image-net.org/image/ILSVRC2015/ILSVRC2015_DET.tar.gz',
    'cbf602d89f2877fa8843392a1ffde03450a18d38'),
    ]
    if not os.path.isdir(args.download_dir):
        makedirs(args.download_dir)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=args.download_dir, overwrite=overwrite, sha1_hash=checksum)
        print(' dataset has already download completed')
        with tarfile.open(filename) as tar:
            tar.extractall(path=args.download_dir)
    if os.path.isdir(os.path.join(args.download_dir, 'ILSVRC2015')):
        os.rename(os.path.join(args.download_dir, 'ILSVRC2015'), os.path.join(args.download_dir, 'ILSVRC'))

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

    frame_crop_base_path = os.path.join(sub_set_crop_path, xml.split('/')[-1].split('.')[0])
    if not os.path.isdir(frame_crop_base_path):
        makedirs(frame_crop_base_path)
    img_path = xml.replace('xml', 'JPEG').replace('Annotations', 'Data')
    im = cv2.imread(img_path)
    avg_chans = np.mean(im, axis=(0, 1))

    for id, object_iter in enumerate(objects):
        bndbox = object_iter.find('bndbox')
        bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
        z, x = crop_like_SiamFC(im, bbox, instance_size=instance_size, padding=avg_chans)
        cv2.imwrite(os.path.join(args.download_dir, frame_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(0, id)), z)
        cv2.imwrite(os.path.join(args.download_dir, frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, id)), x)

def par_crop(args):
    """
    Dataset curation,crop data and transform the format of a label
    """
    crop_path = os.path.join(args.download_dir, './crop{:d}'.format(args.instance_size))
    if not os.path.isdir(crop_path): makedirs(crop_path)
    VID_base_path = os.path.join(args.download_dir, './ILSVRC')
    ann_base_path = os.path.join(VID_base_path, 'Annotations/DET/train/')
    sub_sets = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i')
    for sub_set in sub_sets:
        sub_set_base_path = os.path.join(ann_base_path, sub_set)
        if 'a' == sub_set:
            xmls = sorted(glob.glob(os.path.join(sub_set_base_path, '*', '*.xml')))
        else:
            xmls = sorted(glob.glob(os.path.join(sub_set_base_path, '*.xml')))
        n_imgs = len(xmls)
        sub_set_crop_path = os.path.join(crop_path, sub_set)
        with futures.ProcessPoolExecutor(max_workers=args.num_threads) as executor:
            fs = [executor.submit(crop_xml, args, xml, sub_set_crop_path, args.instance_size) for xml in xmls]
            for i, f in enumerate(futures.as_completed(fs)):
                printProgress(i, n_imgs, prefix=sub_set, suffix='Done ', barLength=80)

def gen_json(args):
    """Format XML and transform json.
       generate train and val json, prepare for tracking dataloader"""
    js = {}
    VID_base_path = os.path.join(args.download_dir, './ILSVRC')
    ann_base_path = os.path.join(VID_base_path, 'Annotations/DET/train/')
    sub_sets = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i')
    for sub_set in sub_sets:
        sub_set_base_path = os.path.join(ann_base_path, sub_set)
        if 'a' == sub_set:
            xmls = sorted(glob.glob(os.path.join(sub_set_base_path, '*', '*.xml')))
        else:
            xmls = sorted(glob.glob(os.path.join(sub_set_base_path, '*.xml')))
        n_imgs = len(xmls)
        for f, xml in enumerate(xmls):
            print('subset: {} frame id: {:08d} / {:08d}'.format(sub_set, f, n_imgs))
            xmltree = ET.parse(xml)
            objects = xmltree.findall('object')

            video = os.path.join(sub_set, xml.split('/')[-1].split('.')[0])

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
    json.dump(train, open(os.path.join(args.download_dir, 'train.json'), 'w'),
              indent=4, sort_keys=True)
    json.dump(val, open(os.path.join(args.download_dir, 'val.json'), 'w'),
              indent=4, sort_keys=True)

def symlink(args):
    """Soft connection in DET"""
    def per_symlink(src, dst):
        """Soft connection"""
        src = os.path.join(args.download_dir, src)
        dst = os.path.join(args.download_dir, dst)
        if not os.path.isdir(dst):
            os.symlink(src, dst)
    per_symlink('ILSVRC/Annotations/DET/train/ILSVRC2013_train', 'ILSVRC/Annotations/DET/train/a')
    per_symlink('ILSVRC/Annotations/DET/train/ILSVRC2014_train_0000', 'ILSVRC/Annotations/DET/train/b')
    per_symlink('ILSVRC/Annotations/DET/train/ILSVRC2014_train_0001', 'ILSVRC/Annotations/DET/train/c')
    per_symlink('ILSVRC/Annotations/DET/train/ILSVRC2014_train_0002', 'ILSVRC/Annotations/DET/train/d')
    per_symlink('ILSVRC/Annotations/DET/train/ILSVRC2014_train_0003', 'ILSVRC/Annotations/DET/train/e')
    per_symlink('ILSVRC/Annotations/DET/train/ILSVRC2014_train_0004', 'ILSVRC/Annotations/DET/train/f')
    per_symlink('ILSVRC/Annotations/DET/train/ILSVRC2014_train_0005', 'ILSVRC/Annotations/DET/train/g')
    per_symlink('ILSVRC/Annotations/DET/train/ILSVRC2014_train_0006', 'ILSVRC/Annotations/DET/train/h')
    per_symlink('ILSVRC/Annotations/DET/val', 'ILSVRC/Annotations/DET/train/i')

    per_symlink('ILSVRC/Data/DET/train/ILSVRC2013_train', 'ILSVRC/Data/DET/train/a')
    per_symlink('ILSVRC/Data/DET/train/ILSVRC2014_train_0000', 'ILSVRC/Data/DET/train/b')
    per_symlink('ILSVRC/Data/DET/train/ILSVRC2014_train_0001', 'ILSVRC/Data/DET/train/c')
    per_symlink('ILSVRC/Data/DET/train/ILSVRC2014_train_0002', 'ILSVRC/Data/DET/train/d')
    per_symlink('ILSVRC/Data/DET/train/ILSVRC2014_train_0003', 'ILSVRC/Data/DET/train/e')
    per_symlink('ILSVRC/Data/DET/train/ILSVRC2014_train_0004', 'ILSVRC/Data/DET/train/f')
    per_symlink('ILSVRC/Data/DET/train/ILSVRC2014_train_0005', 'ILSVRC/Data/DET/train/g')
    per_symlink('ILSVRC/Data/DET/train/ILSVRC2014_train_0006', 'ILSVRC/Data/DET/train/h')
    per_symlink('ILSVRC/Data/DET/val', 'ILSVRC/Data/DET/train/i')


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

