"""
Utility functions for COOT model
"""
import ctypes
import datetime
import logging
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
from typing import Tuple, Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import cuda
import torch.nn.functional as F

EVALKEYS = ["r1", "r5", "r10", "r50", "medr", "meanr", "sum"]
EVALHEADER = "Retriev | R@1   | R@5   | R@10  | R@50  | MeanR |  MedR |    Sum"


def create_dataloader_path(data_root,
                           dataset_name='youcook2',
                           text_feature_name='default',
                           video_feature_name='100m'):
    """create the path to meta file and features

    Args:
        data_root ([PATH]): [Path to the data folder]

    Returns:
        [Dict]: [path to meta data and video/language features]
    """

    meta_data_path = Path(
        os.path.join(data_root, "meta_{}.json".format(video_feature_name)))
    video_feat_path = Path(
        os.path.join(data_root, "video_feat_{}.h5".format(video_feature_name)))
    language_feat_path = Path(
        os.path.join(data_root, "text_{}.h5".format(text_feature_name)))
    meta_text_len_path = Path(
        os.path.join(data_root, "text_lens_{}.json".format(text_feature_name)))

    return {
        "dataset_name": dataset_name,
        "meta_data": meta_data_path,
        "video_feats": video_feat_path,
        "language_feats": language_feat_path,
        "meta_text_len": meta_text_len_path,
    }


def get_csv_header_keys(compute_clip_retrieval):
    """ get CSV header keys"""

    metric_keys = ["ep", "time"]
    prefixes = ["v", "p"]
    if compute_clip_retrieval:
        prefixes += ["c", "s"]
    for prefix in prefixes:
        for key in EVALKEYS:
            metric_keys.append(f"{prefix}-{key}")
    return metric_keys


def expand_segment(num_frames, num_target_frames, start_frame, stop_frame):
    """ expand the segment"""
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
    """ set seed"""
    torch.manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def make_shared_array(np_array: np.ndarray) -> mp.Array:
    """ shared array"""
    flat_shape = int(np.prod(np_array.shape))
    shared_array_base = mp.Array(ctypes.c_float, flat_shape)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(np_array.shape)
    shared_array[:] = np_array[:]
    return shared_array


def compute_indices(num_frames_orig: int, num_frames_target: int,
                    is_train: bool):
    """ compute indices """
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
    """ truncate normal """

    num_examples = 8
    tmp = torch.empty(shape + (num_examples, )).normal_()
    valid = (tmp < limit) & (tmp > -limit)
    _, ind = valid.max(-1, keepdim=True)
    return tmp.gather(-1, ind).squeeze(-1).mul_(std).add_(mean)


def retrieval_results_to_str(results: Dict[str, float], name: str):
    """ retrieval results string """
    return ("{:7s} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:5.1f} | "
            "{:5.1f} | {:6.3f}").format(name, *[results[a] for a in EVALKEYS])


def compute_retr_vid_to_par(video_feat, cap_feat):
    """ compute similarity scores video to paragraph """
    similarity_scores = np.dot(video_feat, cap_feat.T)
    return compute_retrieval_metrics(similarity_scores)


def compute_retr_par_to_vid(video_feat, cap_feat):
    """ compute similarity scores paragraph to video """
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
    """ compare metrics """

    if best is None:
        return True
    threshold = 1e-4
    rel_epsilon = threshold + 1.
    return comparison > best * rel_epsilon


def get_logging_formatter():
    """ logging formatter """
    return logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                             datefmt="%m%d %H%M%S")


def get_timestamp_for_filename():
    """ timestamp"""

    time_split = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    time_split = time_split.replace(":", "_").replace("-", "_")
    return time_split


def get_logger_without_file(name, log_level="INFO") -> logging.Logger:
    """ gett basic logger"""

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
    """Get logger

    Returns:
        logger
    """
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
    """ close logger """

    log_handle_list = list(logger.handlers)
    for i in log_handle_list:
        logger.removeHandler(i)
        i.flush()
        i.close()


def unpack_data(data_dict, use_cuda):
    """unpack data
    """
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


def compute_constrastive_loss(config, contrastive_loss, vid_emb, par_emb,
                              clip_emb, sent_emb, vid_context, par_context):
    """Normalize embeddings and calculate alignment loss in different levels:
     Video-paragraph, clip-sentence, global context

    Args:
        contrastive_loss (loss function): MaxMargingRanking loss
        vid_emb (tensor): video embeddings with shape batch*dim
        par_emb (tensor): paragraph embeddings with shape batch*dim
        clip_emb (tensor): clip embeddings
        sent_emb (tensor): sentence embeddings
        vid_context (tensor): video global context
        par_context (tensor): paragraph global context

    Returns:
        total loss
    """
    vid_context_norm = F.normalize(vid_context)
    clip_emb_norm = F.normalize(clip_emb)
    vid_emb_norm = F.normalize(vid_emb)
    par_context_norm = F.normalize(par_context)
    sent_emb_norm = F.normalize(sent_emb)
    par_emb_norm = F.normalize(par_emb)

    loss = contrastive_loss(vid_emb_norm, par_emb_norm)
    loss += config.CONFIG.TRAIN.LOSS_CONTRASTIVE_CLIP_W * contrastive_loss(
        clip_emb_norm, sent_emb_norm)
    loss += contrastive_loss(vid_context_norm, par_context_norm)
    loss += (contrastive_loss(vid_emb_norm, vid_emb_norm) +
             contrastive_loss(par_emb_norm, par_emb_norm)) / 2
    loss += (contrastive_loss(clip_emb_norm, clip_emb_norm) +
             contrastive_loss(sent_emb_norm, sent_emb_norm)) / 2
    return loss


def compute_cmc_loss(cyc_consistency_loss, loss_weight, clip_emb_reshape,
                     clip_emb_mask, clip_emb_lens, sent_emb_reshape,
                     sent_emb_mask, sent_emb_lens):
    """Calculate the total cycle consistency loss between video clips and paragraph sentences

    Args:
        cyc_consistency_loss (loss function): cycle consistency loss function
        loss_weight (float): weight of loss
        clip_emb_reshape
        clip_emb_mask
        clip_emb_lens
        sent_emb_reshape
        sent_emb_mask
        sent_emb_lens

    Returns:
        total loss
    """
    clip_clip_loss, sent_sent_loss = cyc_consistency_loss(
        clip_emb_reshape, clip_emb_mask, clip_emb_lens, sent_emb_reshape,
        sent_emb_mask, sent_emb_lens)
    loss = loss_weight * (clip_clip_loss + sent_sent_loss)
    return loss
