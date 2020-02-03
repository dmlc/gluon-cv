"""Prepare DET(Object detection) datasets"""
import os
import argparse
import tarfile
from os.path import join, isdir
import glob
import xml.etree.ElementTree as ET
import json
import time
import sys
from gluoncv.utils.filesystem import try_import_cv2
from concurrent import futures
from gluoncv.utils import download, makedirs
cv2 = try_import_cv2()
def parse_args():
    """dataset parameter."""
    parser = argparse.ArgumentParser(
        description='Download DET dataset and prepare for tracking')
    parser.add_argument('--download-dir', type=str, default='~/.mxnet/datasets/DET/',
                        help='dataset directory on disk')
    parser.add_argument('--instanc-size', type=int, default = 511, help='instanc image size')
    parser.add_argument('--num-threads', type=int, default = 12, help='threads number')
    args = parser.parse_args()
    args.download_dir = os.path.expanduser(args.download_dir)
    return args

def download_det(args,overwrite=False):
    url = 'http://image-net.org/image/ILSVRC2015/ILSVRC2015_DET.tar.gz'
    makedirs(args.download_dir)
    filename = download(url, path=args.download_dir, overwrite=overwrite)
    with tarfile.open(filename) as tar:
        tar.extractall(path=args.download_dir)
    os.rename(join(args.download_dir,'ILSVRC2015'),join(args.download_dir,'ILSVRC'))
    

def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar

    function
    ----------
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
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop

def pos_s_2_bbox(pos, s):
    return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
    target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
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


def crop_xml(args, xml, sub_set_crop_path, instanc_size=511):
    xmltree = ET.parse(xml)
    objects = xmltree.findall('object')

    frame_crop_base_path = join(sub_set_crop_path, xml.split('/')[-1].split('.')[0])
    if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)

    img_path = xml.replace('xml', 'JPEG').replace('Annotations', 'Data')
    img_path = join()
    im = cv2.imread(img_path)
    avg_chans = np.mean(im, axis=(0, 1))

    for id, object_iter in enumerate(objects):
        bndbox = object_iter.find('bndbox')
        bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
        z, x = crop_like_SiamFC(im, bbox, instanc_size=instanc_size, padding=avg_chans)
        print(join(args.download_dir,frame_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(0, id)))
        cv2.imwrite(join(args.download_dir,frame_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(0, id)), z)
        cv2.imwrite(join(args.download_dir,frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, id)), x)

def par_crop(args):
    crop_path = join(args.download_dir,'./crop{:d}'.format(args.instanc_size))
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
        #crop_xml(args, xml, sub_set_crop_path, args.instanc_size)
        with futures.ProcessPoolExecutor(max_workers=args.num_threads) as executor:
            fs = [executor.submit(args, crop_xml, xml, sub_set_crop_path, args.instanc_size) for xml in xmls]
            for i, f in enumerate(futures.as_completed(fs)):
                printProgress(i, n_imgs, prefix=sub_set, suffix='Done ', barLength=80)

def gen_json(args):
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

    train = {k:v for (k,v) in js.items() if 'i/' not in k}
    val = {k:v for (k,v) in js.items() if 'i/' in k}
    json.dump(train, open(join(args.download_dir,'train.json'), 'w'),
              indent=4, sort_keys=True)
    json.dump(val, open(join(args.download_dir,'val.json'), 'w'),
              indent=4, sort_keys=True)

def symlink(args):
    os.chdir(args.download_dir)
    os.system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2013_train ILSVRC/Annotations/DET/train/a')
    os.system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0000 ILSVRC/Annotations/DET/train/b')
    os.system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0001 ILSVRC/Annotations/DET/train/c')
    os.system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0002 ILSVRC/Annotations/DET/train/d')
    os.system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0003 ILSVRC/Annotations/DET/train/e')
    os.system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0004 ILSVRC/Annotations/DET/train/f')
    os.system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0005 ILSVRC/Annotations/DET/train/g')
    os.system('ln -sfb $PWD/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0006 ILSVRC/Annotations/DET/train/h')
    os.system('ln -sfb $PWD/ILSVRC/Annotations/DET/val ILSVRC/Annotations/DET/train/i')

    os.system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2013_train ILSVRC/Data/DET/train/a')
    os.system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0000 ILSVRC/Data/DET/train/b')
    os.system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0001 ILSVRC/Data/DET/train/c')
    os.system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0002 ILSVRC/Data/DET/train/d')
    os.system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0003 ILSVRC/Data/DET/train/e')
    os.system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0004 ILSVRC/Data/DET/train/f')
    os.system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0005 ILSVRC/Data/DET/train/g')
    os.system('ln -sfb $PWD/ILSVRC/Data/DET/train/ILSVRC2014_train_0006 ILSVRC/Data/DET/train/h')
    os.system('ln -sfb $PWD/ILSVRC/Data/DET/val ILSVRC/Data/DET/train/i')

def main(args):
    download_det(args)
    symlink(args)
    par_crop(args)
    gen_json(args)

if __name__ == '__main__':
    since = time.time()
    args = parse_args()
    main(args)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

