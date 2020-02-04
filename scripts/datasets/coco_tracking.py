"""this script is used to prepare COCO dataset for tracking,
which is 2017 COCO"""
import argparse
import zipfile
from os.path import join, isdir, expanduser
from concurrent import futures
import sys
import time
import json
import numpy as np
from gluoncv.utils import download, makedirs
from gluoncv.data.mscoco.utils import try_import_pycocotools
from gluoncv.utils.filesystem import try_import_cv2
cv2 = try_import_cv2()

def parse_args():
    """COCO dataset parameter."""
    parser = argparse.ArgumentParser(
        description='Initialize MS COCO dataset.')
    parser.add_argument('--download-dir', type=str, default='~/.mxnet/datasets/coco', help='dataset directory on disk')
    parser.add_argument('--no-download', action='store_true', help='disable automatic download if set')
    parser.add_argument('--overwrite', action='store_true', help='overwrite downloaded files if set, in case they are corrupted')
    parser.add_argument('--instance-size', type=int, default=511,
                        help='instance image size')
    parser.add_argument('--num-threads', type=int, default=12,
                        help='threads number')
    args = parser.parse_args()
    args.download_dir = expanduser(args.download_dir)
    return args

def download_coco(args, overwrite=False):
    """download COCO dataset and Unzip to download_dir"""
    _DOWNLOAD_URLS = [
        ('http://images.cocodataset.org/zips/train2017.zip',
         '10ad623668ab00c62c096f0ed636d6aff41faca5'),
        ('http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
         '8551ee4bb5860311e79dace7e79cb91e432e78b3'),
        ('http://images.cocodataset.org/zips/val2017.zip',
         '4950dc9d00dbe1c933ee0170f5797584351d2a41'),
    ]
    if not isdir(args.download_dir):
        makedirs(args.download_dir)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=args.download_dir, overwrite=overwrite, sha1_hash=checksum)
        with zipfile.ZipFile(filename) as zf:
            zf.extractall(path=args.download_dir)

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


def crop_img(img, anns, set_crop_base_path, set_img_base_path, instance_size=511):
    """
    Dataset curation

    Parameters
    ----------
    img: dic, img
    anns: str, video number
    set_crop_base_path: str, crop result path
    set_img_base_path: str, ori image path
    """
    frame_crop_base_path = join(set_crop_base_path, img['file_name'].split('/')[-1].split('.')[0])
    if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)

    im = cv2.imread('{}/{}'.format(set_img_base_path, img['file_name']))
    avg_chans = np.mean(im, axis=(0, 1))
    for trackid, ann in enumerate(anns):
        rect = ann['bbox']
        bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
        if rect[2] <= 0 or rect[3] <= 0:
            continue
        z, x = crop_like_SiamFC(im, bbox, instance_size=instance_size, padding=avg_chans)
        cv2.imwrite(join(args.download_dir, frame_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(0, trackid)), z)
        cv2.imwrite(join(args.download_dir, frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, trackid)), x)

def crop_coco(args):
    """
    Dataset curation,crop data and transform the format of label

    Parameters
    ----------
    ann_base_path: str, Annotations base path
    """
    crop_path = join(args.download_dir, './crop{:d}'.format(int(args.instance_size)))
    if not isdir(crop_path): makedirs(crop_path)

    for dataType in ['val2017', 'train2017']:
        set_crop_base_path = join(crop_path, dataType)
        set_img_base_path = join(args.download_dir, dataType)

        annFile = '{}/annotations/instances_{}.json'.format(args.download_dir, dataType)
        coco = COCO(annFile)
        n_imgs = len(coco.imgs)
        with futures.ProcessPoolExecutor(max_workers=args.num_threads) as executor:
            fs = [executor.submit(crop_img, coco.loadImgs(id)[0],
                                  coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None)),
                                  set_crop_base_path, set_img_base_path, args.instance_size) for id in coco.imgs]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_imgs, prefix=dataType, suffix='Done ', barLength=40)
    print('done')

def gen_json(args):
    """transform json and generate train and val json, prepare for tracking dataloader"""
    for dataType in ['val2017', 'train2017']:
        dataset = dict()
        annFile = '{}/annotations/instances_{}.json'.format(args.download_dir, dataType)
        coco = COCO(annFile)
        n_imgs = len(coco.imgs)
        for n, img_id in enumerate(coco.imgs):
            print('subset: {} image id: {:04d} / {:04d}'.format(dataType, n, n_imgs))
            img = coco.loadImgs(img_id)[0]
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)
            video_crop_base_path = join(dataType, img['file_name'].split('/')[-1].split('.')[0])

            if len(anns) > 0:
                dataset[video_crop_base_path] = dict()

            for trackid, ann in enumerate(anns):
                rect = ann['bbox']
                c = ann['category_id']
                bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
                if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                    continue
                dataset[video_crop_base_path]['{:02d}'.format(trackid)] = {'000000': bbox}

        print('save json (dataset), please wait 20 seconds~')
        json.dump(dataset, open('{}.json'.format(dataType), 'w'), indent=4, sort_keys=True)
        print('done!')

def main(args):
    # download COCO dataset
    # download_coco(args, overwrite=args.overwrite)
    print('COCO dataset json has already generat completed')
    # crop COCO dataset for prepare for tracking
    crop_coco(args)
    print('COCO dataset has already crop completed')
    # generat COCO json for prepare for tracking
    gen_json(args)
    print('COCO dataset has already generat completed')

if __name__ == '__main__':
    try_import_pycocotools()
    from pycocotools.coco import COCO
    since = time.time()
    args = parse_args()
    main(args)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
