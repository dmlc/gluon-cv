"""Initialize ImageNet dataset. http://www.image-net.org/challenges/LSVRC/2012/
You need to download the massive data files by your self from:
http://www.image-net.org/download-images since it use only public for non-commercial use.
An account is required to obtain valid download links.
Train: ILSVRC2012_img_train.tar(138G), validation: ILSVRC2012_img_val.tar(6.3G)
We make symbolic link to '~/.mxnet/datasets/imagenet' so users can use it our-of-box in the future.
"""
import os
import argparse
import tarfile
from tqdm import tqdm
from gluonvision.utils import makedirs
from mxnet.gluon.utils import check_sha1

_TARGET_DIR = os.path.expanduser('~/.mxnet/datasets/imagenet')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize imagenet dataset.',
        epilog='Example: python setup_imagenet.py ~/datasets/VOCdevkit --download',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', required=True, help='dataset directory on disk')
    parser.add_argument('--download-dir', type=str, default='',
                        help='directory of downloaded *.tar files')
    parser.add_argument('--check-sha1', action='store_true',
                        help='check sha1 checksum before untar, since there is '
                        'risk of corrupted files. Use with cautious because this can be very slow.')
    args = parser.parse_args()
    return args

def untar_imagenet_train(filename, target_dir, checksum=False):
    train_dir = os.path.join(os.path.expanduser(target_dir), 'train')
    if checksum:
        if not check_sha1(filename, '43eda4fe35c1705d6606a6a7a633bc965d194284'):
            raise RuntimeError("Corrupted training file: {}".format(filename))

    makedirs(train_dir)
    with tarfile.open(filename) as tar:
        print("Extracting train data, this may take very long time depending on I/O perf.")
        files = tar.getnames()
        assert len(files) == 1000
        with tqdm(total=1000) as pbar:
            for obj in tar:
                if not obj:
                    break
                tar.extract(obj, train_dir)
                iname = os.path.join(train_dir, obj.name)
                with tarfile.open(iname) as itar:
                    dst_dir = os.path.splitext(iname)[0]
                    makedirs(dst_dir)
                    itar.extractall(dst_dir)
                # remove separate tarfiles
                os.remove(iname)
                pbar.update(1)

def untar_imagenet_val(filename, target_dir, checksum=False):
    val_dir = os.path.join(os.path.expanduser(target_dir), 'val')
    if checksum:
        if not check_sha1(filename, '5f3f73da3395154b60528b2b2a2caf2374f5f178'):
            raise RuntimeError("Corrupted training file: {}".format(filename))

    makedirs(val_dir)
    with tarfile.open(filename) as tar:
        tar.extractall(val_dir)

def symlink_val(val_dir, dst_dir):
    """Symlink individual validation images to 1000 directories."""
    val_dir = os.path.expanduser(val_dir)
    dst_dir = os.path.expanduser(dst_dir)
    if os.path.isdir(os.path.join(val_dir, 'n04429376')):
        # already organzied into 1000 directories, just symlink parent dir
        os.symlink(val_dir, dst_dir)
        return
    import pickle
    val_maps_file = os.path.join(os.path.dirname(__file__), 'imagenet_val_maps.pkl')
    with open(val_maps_file, 'rb') as f:
        dirs, mappings = pickle.load(f)
    assert len(dirs) == 1000, "Require 1000 dir names"
    assert len(mappings) == 50000, "Require 50000 image->dir mappings"

    for d in dirs:
        makedirs(os.path.join(dst_dir, d))

    for m in mappings:
        os.symlink(os.path.join(val_dir, m[0]), os.path.join(dst_dir, m[1], m[0]))

if __name__ == '__main__':
    args = parse_args()
    path = os.path.expanduser(args.path)
    if not os.path.isdir(path) or not os.path.isdir(os.path.join(path, 'train')) \
        or not os.path.isdir(os.path.join(path, 'train', 'n04429376')) \
        or not os.path.isdir(os.path.join(path, 'val')):
        if not args.download_dir:
            raise ValueError(('{} is not a valid ImageNet directory, make sure it is present.'
                              ' Or you can point "--download-dir" to tar files'.format(path)))
        else:
            train_tarfile = os.path.join(args.download_dir, 'ILSVRC2012_img_train.tar')
            untar_imagenet_train(train_tarfile, path, args.check_sha1)
            val_tarfile = os.path.join(args.download_dir, 'ILSVRC2012_img_val.tar')
            untar_imagenet_val(val_tarfile, path, args.check_sha1)

    # make symlink
    if os.path.isdir(_TARGET_DIR):
        os.remove(_TARGET_DIR)
    makedirs(_TARGET_DIR)
    os.symlink(os.path.join(args.path, 'train'), os.path.join(_TARGET_DIR, 'train'))
    symlink_val(os.path.join(args.path, 'val'), os.path.join(_TARGET_DIR, 'val'))
