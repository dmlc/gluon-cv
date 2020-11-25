import csv
import ctypes
import datetime
from easydict import EasyDict
import logging
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path
from typing import Union, Tuple, Dict
import yaml

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import cuda

EVALKEYS = ["r1", "r5", "r10", "r50", "medr", "meanr", "sum"]
EVALHEADER = "Retriev | R@1   | R@5   | R@10  | R@50  | MeanR |  MedR |    Sum"


def create_dataloader_path(data_root,
                           shot_per_group,
                           dataset_name,
                           text_feature_name='default',
                           video_feature_name='howto_h100m'):
    """create the path to meta file and features

    Args:
        data_root ([PATH]): [Path to the data folder]
        shot_per_group ([Int]): [number of shots (clips) per group (video)]

    Returns:
        [Dict]: [path to meta data and video/language features]
    """

    meta_data_path = Path(
        os.path.join(data_root, "meta",
                     "meta_group{}.json".format(shot_per_group)))
    video_feat_path = Path(
        os.path.join(data_root, "group{}".format(shot_per_group),
                     "video_features", "{}.h5".format(video_feature_name)))
    language_feat_path = Path(
        os.path.join(data_root, "group{}".format(shot_per_group),
                     "language_features",
                     "text_{}.h5".format(text_feature_name)))
    meta_text_len_path = Path(
        os.path.join(data_root, "group{}".format(shot_per_group),
                     "language_features",
                     "text_lens_{}.json".format(text_feature_name)))

    return {
        "meta_data": meta_data_path,
        "video_feats": video_feat_path,
        "language_feats": language_feat_path,
        "meta_text_len": meta_text_len_path,
        "dataset_name": dataset_name
    }


def get_csv_header_keys(compute_clip_retrieval):
    metric_keys = ["ep", "time"]
    prefixes = ["v", "p"]
    if compute_clip_retrieval:
        prefixes += ["c", "s"]
    for prefix in prefixes:
        for key in EVALKEYS:
            metric_keys.append(f"{prefix}-{key}")
    return metric_keys


def print_csv_results(csv_file: str, cfg: EasyDict, print_fn=print):
    metric_keys = get_csv_header_keys(cfg.training.compute_clip_retrieval)
    with Path(csv_file).open("rt", encoding="utf8") as fh:
        reader = csv.DictReader(fh, metric_keys)
        line_data = [line for line in reader][1:]
        for line in line_data:
            for key, val in line.items():
                line[key] = float(val)
    if cfg.training.det_best_field == "val_score_at_1":
        relevant_field = [line["v-r1"] + line["p-r1"] for line in line_data]
    elif cfg.training.det_best_field == "val_clip_score_at_1":
        relevant_field = [line["c-r1"] + line["s-r1"] for line in line_data]
    else:
        raise NotImplementedError
    best_epoch = np.argmax(relevant_field)

    def get_res(search_key):
        results = {}
        for key_, val_ in line_data[best_epoch].items():
            if key_[:2] == f"{search_key}-":
                results[key_[2:]] = float(val_)
        return results

    print_fn(f"Total epochs {len(line_data)}. "
             f"Results from best epoch {best_epoch}:")
    print_fn(EVALHEADER)
    print_fn(retrieval_results_to_str(get_res("p"), "Par2Vid"))
    print_fn(retrieval_results_to_str(get_res("v"), "Vid2Par"))
    if cfg.training.compute_clip_retrieval:
        print_fn(retrieval_results_to_str(get_res("s"), "Sen2Cli"))
        print_fn(retrieval_results_to_str(get_res("c"), "Cli2Sen"))


def expand_segment(num_frames, num_target_frames, start_frame, stop_frame):
    num_frames_seg = stop_frame - start_frame + 1
    changes = False
    if num_target_frames > num_frames:
        num_target_frames = num_frames
    if num_frames_seg < num_target_frames:
        while True:
            if start_frame > 0:
                start_frame -= 1
                num_frames_seg += 1
                changes = True
            if num_frames_seg == num_target_frames:
                break
            if stop_frame < num_frames - 1:
                stop_frame += 1
                num_frames_seg += 1
                changes = True
            if num_frames_seg == num_target_frames:
                break
    return start_frame, stop_frame, changes


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_config(file: Union[str, Path]) -> EasyDict:
    with Path(file).open("rt", encoding="utf8") as fh:
        config = yaml.load(fh, Loader=yaml.Loader)
    cfg = EasyDict(config)
    # model symmetry
    for check_network in ["text_pooler", "text_sequencer"]:
        if getattr(cfg, check_network).name == "same":
            setattr(cfg, check_network,
                    getattr(cfg,
                            getattr(cfg, check_network).same_as))
    return cfg


def dump_config(cfg: EasyDict, file: Union[str, Path]) -> None:
    with Path(file).open("wt", encoding="utf8") as fh:
        yaml.dump(cfg, fh, Dumper=yaml.Dumper)


def print_config(cfg: EasyDict, level=0) -> None:
    for key, val in cfg.items():
        if isinstance(val, EasyDict):
            print("     " * level, str(key), sep="")
            print_config(val, level=level + 1)
        else:
            print("    " * level, f"{key} - f{val} ({type(val)})", sep="")


def make_shared_array(np_array: np.ndarray) -> mp.Array:
    flat_shape = int(np.prod(np_array.shape))
    shared_array_base = mp.Array(ctypes.c_float, flat_shape)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(np_array.shape)
    shared_array[:] = np_array[:]
    return shared_array


def compute_indices(num_frames_orig: int, num_frames_target: int,
                    is_train: bool):
    def round_half_down(array: np.ndarray) -> np.ndarray:
        return np.ceil(array - 0.5)

    if is_train:
        # random sampling during training
        start_points = np.linspace(0,
                                   num_frames_orig,
                                   num_frames_target,
                                   endpoint=False)
        start_points = round_half_down(start_points).astype(int)
        offsets = start_points[1:] - start_points[:-1]
        np.random.shuffle(offsets)
        last_offset = num_frames_orig - np.sum(offsets)
        offsets = np.concatenate([offsets, np.array([last_offset])])
        new_start_points = np.cumsum(offsets) - offsets[0]
        offsets = np.roll(offsets, -1)
        random_offsets = offsets * np.random.rand(num_frames_target)
        indices = new_start_points + random_offsets
        indices = np.floor(indices).astype(int)
        return indices
    # center sampling during validation
    start_points = np.linspace(0,
                               num_frames_orig,
                               num_frames_target,
                               endpoint=False)
    offset = num_frames_orig / num_frames_target / 2
    indices = start_points + offset
    indices = np.floor(indices).astype(int)
    return indices


def truncated_normal_fill(shape: Tuple[int],
                          mean: float = 0,
                          std: float = 1,
                          limit: float = 2) -> torch.Tensor:
    num_examples = 8
    tmp = torch.empty(shape + (num_examples, )).normal_()
    valid = (tmp < limit) & (tmp > -limit)
    _, ind = valid.max(-1, keepdim=True)
    return tmp.gather(-1, ind).squeeze(-1).mul_(std).add_(mean)


def retrieval_results_to_str(results: Dict[str, float], name: str):
    return ("{:7s} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:5.1f} | "
            "{:5.1f} | {:6.3f}").format(name, *[results[a] for a in EVALKEYS])


def compute_retr_vid_to_par(video_feat, cap_feat):
    similarity_scores = np.dot(video_feat, cap_feat.T)
    return compute_retrieval_metrics(similarity_scores)


def compute_retr_par_to_vid(video_feat, cap_feat):
    similarity_scores = np.dot(cap_feat, video_feat.T)
    return compute_retrieval_metrics(similarity_scores)


def compute_retrieval_metrics(dot_product):
    """ Compute the retrieval performance

    Args:
        dot_product (similarity of embeddings X1 and X2): dot_product(X1, X2)

    Returns:
        Retrieval evaluation metrics such as R1, R5 and so on.
    """

    sort_similarity = np.sort(-dot_product, axis=1)
    diag_similarity = np.diag(-dot_product)
    diag_similarity = diag_similarity[:, np.newaxis]
    ranks = sort_similarity - diag_similarity
    ranks = np.where(ranks == 0)
    ranks = ranks[1]

    report_dict = dict()
    report_dict['r1'] = float(np.sum(ranks == 0)) / len(ranks)
    report_dict['r5'] = float(np.sum(ranks < 5)) / len(ranks)
    report_dict['r10'] = float(np.sum(ranks < 10)) / len(ranks)
    report_dict['r50'] = float(np.sum(ranks < 50)) / len(ranks)
    report_dict['medr'] = np.median(ranks) + 1
    report_dict['meanr'] = ranks.mean()
    report_dict[
        'sum'] = report_dict['r1'] + report_dict['r5'] + report_dict['r50']

    return report_dict, ranks


def compare_metrics(comparison, best):
    if best is None:
        return True
    threshold = 1e-4
    rel_epsilon = threshold + 1.
    return comparison > best * rel_epsilon

def get_logging_formatter():
    return logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                             datefmt="%m%d %H%M%S")


def get_timestamp_for_filename():
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    return ts


def get_logger_without_file(name, log_level="INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(get_logging_formatter())
    logger.addHandler(strm_hdlr)
    return logger


def get_logger(logdir,
               name,
               filename="run",
               log_level="INFO",
               log_file=True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    formatter = get_logging_formatter()
    if log_file:
        file_path = Path(logdir) / "{}_{}.log".format(
            filename,
            str(datetime.datetime.now()).split(".")[0].replace(
                " ", "_").replace(":", "_").replace("-", "_"))
        file_hdlr = logging.FileHandler(str(file_path))
        file_hdlr.setFormatter(formatter)
        logger.addHandler(file_hdlr)
    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)
    logger.addHandler(strm_hdlr)
    logger.propagate = False
    return logger


def close_logger(logger: logging.Logger):
    x = list(logger.handlers)
    for i in x:
        logger.removeHandler(i)
        i.flush()
        i.close()

def unpack_data(data_dict, use_cuda):
    def to_device(x):
        if use_cuda and isinstance(x, torch.Tensor):
            return x.cuda(non_blocking=True)
        return x

    return [
        to_device(data_dict[a])
        for a in ("vid_id", "vid_frames", "vid_frames_mask", "vid_frames_len",
                  "par_cap_vectors", "par_cap_mask", "par_cap_len", "clip_num",
                  "clip_frames", "clip_frames_len", "clip_frames_mask",
                  "sent_num", "sent_cap_vectors", "sent_cap_mask",
                  "sent_cap_len")
    ]