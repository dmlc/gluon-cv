# pylint: disable=missing-function-docstring, line-too-long, consider-using-with
"""
Utility functions, misc
"""
import os
import sys
import time
import functools
import logging
from contextlib import contextmanager
from functools import wraps

import numpy as np
import torch.nn as nn
import torch


def read_labelmap(labelmap_file):
    """Read label map and class ids."""

    labelmap = []
    class_ids = set()
    name = ""
    class_id = ""
    with open(labelmap_file, 'r') as f:
        for line in f:
            if line.startswith("  name:"):
                name = line.split('"')[1]
            elif line.startswith("  id:") or line.startswith("  label_id:"):
                class_id = int(line.strip().split(" ")[-1])
                labelmap.append({"id": class_id, "name": name})
                class_ids.add(class_id)
    return labelmap, class_ids

# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")

def build_log_dir(cfg):
    # create base log directory
    if cfg.CONFIG.LOG.EXP_NAME == 'use_time':
        cfg.CONFIG.LOG.EXP_NAME = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # setup log file and stdout
    logger = logging.getLogger("gluoncv")
    log_level = logging.getLevelName(cfg.CONFIG.LOG.LEVEL)
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    logger.setLevel(log_level)
    logger.propagate = False
    # file logger, enabled for all workers
    log_filename = os.path.join(log_path, cfg.CONFIG.LOG.LOG_FILENAME)
    if cfg.DDP_CONFIG.WORLD_RANK > 0:
        log_filename += f".rank{cfg.DDP_CONFIG.WORLD_RANK}"
    fh = logging.StreamHandler(_cached_log_stream(log_filename))
    fh.setLevel(log_level)
    fh.setFormatter(plain_formatter)
    logger.addHandler(fh)
    if cfg.DDP_CONFIG.WORLD_RANK == 0:
        # stdout in rank 0 only
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(log_level)
        logger.addHandler(ch)

    # dump config file
    with open(os.path.join(log_path, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())

    # create tensorboard saving directory
    tb_logdir = os.path.join(log_path, cfg.CONFIG.LOG.LOG_DIR)
    if not os.path.exists(tb_logdir):
        os.makedirs(tb_logdir)

    # create checkpoint saving directory
    ckpt_logdir = os.path.join(log_path, cfg.CONFIG.LOG.SAVE_DIR)
    if not os.path.exists(ckpt_logdir):
        os.makedirs(ckpt_logdir)

    return tb_logdir


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_mAP(output, target):
    import torchnet.meter as meter
    mtr = meter.mAPMeter()
    mtr.add(output, target)
    ap = mtr.value()
    return ap


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def per_class_error(output, target, num_classes):
    per_class_acc = []
    for i in range(num_classes):
        index = np.where(target == i)
        diff = output[index] - target[index]
        corr = np.where(diff == 0)[0].shape[0] / 1.0
        total = index[0].shape[0] / 1.0
        print(corr, total)
        per_class_acc.append(corr / total)

    return per_class_acc


class ProgressMeter(object):
    """Default PyTorch pogress meter"""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def constant_init(module, val, bias=0):
    # Code adapted from https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/weight_init.py
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    # Code adapted from https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/weight_init.py
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if "CUDA out of memory. " in str(e):
            pass
        else:
            raise


def retry_if_cuda_oom(func):
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.

    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.

    Args:
        func: a stateless callable that takes tensor-like objects as arguments

    Returns:
        a callable which retries `func` if OOM is encountered.

    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU

    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x.to(device="cpu")
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__)
        logger.info("Attempting to copy inputs of {} to CPU due to CUDA OOM".format(str(func)))
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapped
