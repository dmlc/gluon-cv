"""
Utility functions for model
"""
import os
import hashlib
import requests
from tqdm import tqdm

import torch


def deploy_model(model, cfg):
    """
    Deploy model to multiple GPUs for DDP training.
    """
    if cfg.DDP_CONFIG.DISTRIBUTED:
        if cfg.DDP_CONFIG.GPU is not None:
            torch.cuda.set_device(cfg.DDP_CONFIG.GPU)
            model.cuda(cfg.DDP_CONFIG.GPU)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[cfg.DDP_CONFIG.GPU],
                                                              find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif cfg.DDP_CONFIG.GPU is not None:
        torch.cuda.set_device(cfg.DDP_CONFIG.GPU)
        model = model.cuda(cfg.DDP_CONFIG.GPU)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    return model


def load_model(model, cfg, load_fc=True):
    """
    Load pretrained model weights.
    """
    if os.path.isfile(cfg.CONFIG.MODEL.PRETRAINED_PATH):
        print("=> loading checkpoint '{}'".format(cfg.CONFIG.MODEL.PRETRAINED_PATH))
        if cfg.DDP_CONFIG.GPU is None:
            checkpoint = torch.load(cfg.CONFIG.MODEL.PRETRAINED_PATH)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(cfg.DDP_CONFIG.GPU)
            checkpoint = torch.load(cfg.CONFIG.MODEL.PRETRAINED_PATH, map_location=loc)
        model_dict = model.state_dict()
        if not load_fc:
            del model_dict['module.fc.weight']
            del model_dict['module.fc.bias']
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
        unused_dict = {k: v for k, v in checkpoint['state_dict'].items() if not k in model_dict}
        not_found_dict = {k: v for k, v in model_dict.items() if not k in checkpoint['state_dict']}
        print("unused model layers:", unused_dict.keys())
        print("not found layers:", not_found_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(cfg.CONFIG.MODEL.PRETRAINED_PATH, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(cfg.CONFIG.MODEL.PRETRAINED_PATH))

    return model, None


def save_model(model, optimizer, epoch, cfg):
    # pylint: disable=line-too-long
    """
    Save trained model weights.
    """
    model_save_dir = os.path.join(cfg.CONFIG.LOG.BASE_PATH,
                                  cfg.CONFIG.LOG.EXP_NAME,
                                  cfg.CONFIG.LOG.SAVE_DIR)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    ckpt_name = "f{}_s{}_ckpt_epoch{}.pth".format(cfg.CONFIG.DATA.CLIP_LEN, cfg.CONFIG.DATA.FRAME_RATE, epoch)
    checkpoint = os.path.join(model_save_dir, ckpt_name)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc1': None,
        'optimizer': optimizer.state_dict(),
    }, filename=checkpoint)


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    sha1_file = sha1.hexdigest()
    l = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]


def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...'%(fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s"%url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None: # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname
