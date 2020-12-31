import json
from collections import OrderedDict
from pathlib import Path
from typing import List
import re

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from gluoncv.torch.utils import coot_utils


def load_h5file(h5_path):
    import h5py
    return h5py.File(h5_path, "r")

class BertTextFeatureLoader:
    def __init__(
            self, dataset_path_dict, ids: List[str], preload=True):
        self.h5_path = dataset_path_dict["language_feats"]
        lens_file = dataset_path_dict["meta_text_len"]
        self.lens = json.load(lens_file.open("rt", encoding="utf8"))
        self.cached_data = None
        if preload:
            h5file = load_h5file(self.h5_path)
            self.cached_data = {}
            for id_ in tqdm(ids, desc="preload text"):
                np_array = h5file[id_]
                shared_array = coot_utils.make_shared_array(np_array)
                self.cached_data[id_] = shared_array
            h5file.close()

    def __getitem__(self, id_):
        lens = self.lens[id_]
        if self.cached_data is None:
            h5file = load_h5file(self.h5_path)
            features = np.array(h5file[id_])
            h5file.close()
            return features, lens
        return self.cached_data[id_], lens


class ActivityNetVideoFeatureLoader:
    def __init__(self, dataset_path: Path, ids: List[str], preload: bool):
        self.dataset_path = Path(dataset_path)
        self.features_path = (dataset_path / "features" /
                              "ICEP_V3_global_pool_skip_8_direct_resize")
        self.cached_data = None
        if preload:
            self.cached_data = {}
            for id_ in tqdm(ids, desc="preload videos"):
                np_array = self.load_from_file(id_)
                shared_array = coot_utils.make_shared_array(np_array)
                self.cached_data[id_] = shared_array

    def __getitem__(self, id_):
        if self.cached_data is None:
            return self.load_from_file(id_)
        else:
            return self.cached_data[id_]

    def load_from_file(self, id_):
        return np.load(str(self.features_path / f"{id_}.npz"))[
            "frame_scores"].squeeze(1).squeeze(2).squeeze(2)


class Youcook2VideoFeatureLoader:
    def __init__(
            self, dataset_path_dict, ids: List[str],
            preload: bool):
        self.h5_path = dataset_path_dict["video_feats"]
        self.cached_data = None
        if preload:
            self.cached_data = {}
            h5file = load_h5file(self.h5_path)
            for id_ in tqdm(ids, desc="preload videos"):
                np_array = h5file[id_]
                shared_array = coot_utils.make_shared_array(np_array)
                self.cached_data[id_] = shared_array

    def __getitem__(self, id_):
        if self.cached_data is None:
            h5file = load_h5file(self.h5_path)
            features = np.array(h5file[id_])
            h5file.close()
            return features
        else:
            return self.cached_data[id_]


class VideoDatasetFeatures(data.Dataset):
    def __init__(
            self, dataset_path_dict,
            split: str, max_frames: int, is_train: bool,
            preload_vid_feat: bool, preload_text_feat: bool,
            frames_noise: float):
        self.frames_noise = frames_noise
        self.split = split
        self.max_frames = max_frames
        self.is_train = is_train
        meta_file = dataset_path_dict["meta_data"]
        
        self.vids_dict = json.load(meta_file.open("rt", encoding="utf8"),
                                   object_pairs_hook=OrderedDict)
        self.ids = [key for key, val in self.vids_dict.items(
        ) if val["split"] == self.split]
        print("init dataset {} split {} length {} ".format(dataset_path_dict["dataset_name"], split, len(self)))

        if dataset_path_dict["dataset_name"]  == "activitynet":
            self.text_data = BertTextFeatureLoader(
                dataset_path, self.ids, dataset_features,
                preload_text_feat)
            self.preproc_par_fn = self.preprocess_bert_paragraph
            self.vid_data = ActivityNetVideoFeatureLoader(
                dataset_path, self.ids, preload_vid_feat)
        elif dataset_path_dict["dataset_name"] == "youcook2":
            self.preproc_par_fn = self.preprocess_bert_paragraph
            self.text_data = BertTextFeatureLoader(
                dataset_path_dict, self.ids, preload_text_feat)
            self.vid_data = Youcook2VideoFeatureLoader(
                dataset_path_dict, self.ids, preload_vid_feat)
        else:
            raise NotImplementedError


    def preprocess_bert_sentence(self, sentence_str: str) -> List[str]:
        if sentence_str[-1] == ".":
            sentence_str = sentence_str[:-1]
        sentence_str = sentence_str.replace(". ", " [SEP] ")
        sentence_str += " [SEP] "
        sentence_str = re.sub(r"\s+", " ", sentence_str).strip()
        words = sentence_str.split(" ")
        return words

    def preprocess_bert_paragraph(self,
            paragraph: List[str]) -> List[List[str]]:
        new_paragraph = []
        for i, sentence in enumerate(paragraph):
            new_sentence = []
            if i == 0:
                new_sentence.append("[CLS]")
            preproc_sentence = self.preprocess_bert_sentence(sentence)
            for word in preproc_sentence:
                new_sentence.append(word.strip())
            new_paragraph.append(new_sentence)
        return new_paragraph


    def get_frames_from_video(
            self, vid_id, indices=None, num_frames=None):
        vid_dict = self.vids_dict[vid_id]
        vid_len = vid_dict["num_frames"]
        if num_frames is not None:
            indices = coot_utils.compute_indices(
                vid_len, num_frames, self.is_train)
        frames = self.vid_data[vid_id][indices]
        return frames

    def get_frames_from_segment(
            self, vid_id, seg_num, num_frames):
        vid_dict = self.vids_dict[vid_id]
        seg = vid_dict["segments"][seg_num]
        start_frame = seg["start_frame"]
        seg_len = seg["num_frames"]
        indices = coot_utils.compute_indices(seg_len, num_frames, self.is_train)
        indices += start_frame
        frames = self.get_frames_from_video(vid_id, indices)
        return frames

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        vid_id = self.ids[index]
        vid_dict = self.vids_dict[vid_id]
        clip_num = len(vid_dict["segments"])
        sent_num = len(vid_dict["segments"])

        # load video frames
        vid_frames_len = vid_dict["num_frames"]
        if vid_frames_len > self.max_frames:
            vid_frames_len = self.max_frames
        vid_frames = torch.tensor(self.get_frames_from_video(
            vid_id, num_frames=vid_frames_len))
        vid_frames_len = int(vid_frames.shape[0])
        if self.frames_noise != 0:
            vid_frames_noise = coot_utils.truncated_normal_fill(
                vid_frames.shape, std=self.frames_noise)
            vid_frames += vid_frames_noise

        # load segment frames
        clip_frames_list = []
        clip_frames_len_list = []
        for i, seg in enumerate(vid_dict["segments"]):
            c_num_frames = seg["num_frames"]
            if c_num_frames > self.max_frames:
                c_num_frames = self.max_frames
            c_frames = self.get_frames_from_segment(
                vid_id, i, num_frames=c_num_frames)
            c_frames = torch.tensor(c_frames)
            if self.frames_noise != 0:
                clip_frames_noise = coot_utils.truncated_normal_fill(
                    c_frames.shape, std=self.frames_noise)
                c_frames += clip_frames_noise
            clip_frames_list.append(c_frames)
            clip_frames_len_list.append(c_frames.shape[0])

        # load text
        seg_narrations = []
        for seg in vid_dict["segments"]:
            seg_narr = seg["narration"]
            if seg_narr is None:
                seg_narr = "undefined"
                print("WARNING: Undefined text tokens "
                      "(no narration data, is this a test set?)")
            seg_narrations.append(seg_narr)
        list_of_list_of_words = self.preproc_par_fn(seg_narrations)

        # load precomputed text features
        par_cap_vectors, sent_cap_len_list = self.text_data[vid_id]
        par_cap_len = int(par_cap_vectors.shape[0])
        par_cap_vectors = torch.tensor(par_cap_vectors).float()

        # split paragraph features into sentences
        sent_cap_vectors_list = []
        pointer = 0
        for i, sent_cap_len in enumerate(sent_cap_len_list):
            sent_cap_vectors = par_cap_vectors[
                               pointer:pointer + sent_cap_len, :]
            sent_cap_vectors_list.append(sent_cap_vectors)
            pointer += sent_cap_len

        return {
            "vid_id": vid_id,
            "data_words": list_of_list_of_words,
            "vid_frames": vid_frames,
            "vid_frames_len": vid_frames_len,
            "par_cap_vectors": par_cap_vectors,
            "par_cap_len": par_cap_len,
            "clip_num": clip_num,
            "sent_num": sent_num,
            "clip_frames_list": clip_frames_list,
            "clip_frames_len_list": clip_frames_len_list,
            "sent_cap_len_list": sent_cap_len_list,
            "sent_cap_vectors_list": sent_cap_vectors_list
        }

    def collate_fn(self, data_batch):
        def get_data(key):
            return [d[key] for d in data_batch]

        batch_size = len(data_batch)

        # collate video frames
        list_vid_frames = get_data("vid_frames")
        list_vid_frames_len = get_data("vid_frames_len")
        vid_feature_dim = list_vid_frames[0].shape[-1]
        vid_frames_len = torch.tensor(list_vid_frames_len).long()
        vid_frames_max_seq_len = int(vid_frames_len.max().numpy())
        vid_frames = torch.zeros(
            batch_size, vid_frames_max_seq_len, vid_feature_dim).float()
        vid_frames_mask = torch.zeros(batch_size, vid_frames_max_seq_len)
        for batch, (seq_len, item) in enumerate(zip(
                list_vid_frames_len, list_vid_frames)):
            vid_frames[batch, :seq_len] = item
            vid_frames_mask[batch, :seq_len] = 1

        # collate paragraph features
        list_par_cap_len = get_data("par_cap_len")
        list_par_cap_vectors = get_data("par_cap_vectors")
        par_feature_dim = list_par_cap_vectors[0].shape[-1]
        par_cap_len = torch.tensor(list_par_cap_len).long()
        par_cap_max_len = int(par_cap_len.max().numpy())
        par_cap_vectors = torch.zeros(
            batch_size, par_cap_max_len, par_feature_dim).float()
        par_cap_mask = torch.zeros(batch_size, par_cap_max_len)
        for batch, (seq_len, item) in enumerate(
                zip(list_par_cap_len, list_par_cap_vectors)):
            par_cap_vectors[batch, :seq_len, :] = item
            par_cap_mask[batch, :seq_len] = 1

        # collate clip frames
        list_clip_num = get_data("clip_num")
        clip_num = torch.tensor(list_clip_num).long()
        total_clip_num = int(np.sum(list_clip_num))
        list_clip_frames_len_list = get_data("clip_frames_len_list")
        clip_frames_max_len = int(np.max(
            [np.max(len_single) for len_single in list_clip_frames_len_list]))
        clip_frames = torch.zeros((
            total_clip_num, clip_frames_max_len, vid_feature_dim)).float()
        clip_frames_mask = torch.zeros(
            (total_clip_num, clip_frames_max_len))
        list_clip_frames_list = get_data("clip_frames_list")
        clip_frames_len = []
        c_num = 0
        for batch, clip_frames_list in enumerate(list_clip_frames_list):
            for i, clip_frames_item in enumerate(clip_frames_list):
                clip_frames_len_item = int(clip_frames_item.shape[0])
                clip_frames[c_num, :clip_frames_len_item, :] =\
                    clip_frames_item
                clip_frames_mask[c_num, :clip_frames_len_item] = 1
                clip_frames_len.append(clip_frames_len_item)
                c_num += 1
        clip_frames_len = torch.tensor(clip_frames_len).long()

        # collate sentence features
        list_sent_num = get_data("sent_num")
        sent_num = torch.tensor(list_sent_num).long()
        total_sent_num = int(np.sum(list_sent_num))
        list_sent_cap_len_list = get_data("sent_cap_len_list")
        sent_cap_max_len = int(np.max(
            [np.max(len_single) for len_single in list_sent_cap_len_list]))
        sent_cap_len = []
        sent_cap_mask = torch.zeros(
            (total_sent_num, sent_cap_max_len)).long()
        cap_feature_dim = list_par_cap_vectors[0].shape[-1]
        sent_cap_vectors = torch.zeros(
            (total_sent_num, sent_cap_max_len, cap_feature_dim))
        c_num = 0
        for batch, sent_cap_len_list in enumerate(
                list_sent_cap_len_list):
            pointer = 0
            for sent_cap_len_item in sent_cap_len_list:
                sent_cap_vectors[c_num, :sent_cap_len_item] =\
                    par_cap_vectors[
                    batch, pointer:pointer + sent_cap_len_item]
                sent_cap_mask[c_num, :sent_cap_len_item] = 1
                sent_cap_len.append(sent_cap_len_item)
                c_num += 1
                pointer += sent_cap_len_item
        sent_cap_len = torch.tensor(sent_cap_len).long()

        return {
            "vid_frames": vid_frames,
            "vid_frames_mask": vid_frames_mask,
            "vid_frames_len": vid_frames_len,
            "par_cap_vectors": par_cap_vectors,
            "par_cap_mask": par_cap_mask,
            "par_cap_len": par_cap_len,
            "clip_num": clip_num,
            "clip_frames": clip_frames,
            "clip_frames_len": clip_frames_len,
            "clip_frames_mask": clip_frames_mask,
            "sent_num": sent_num,
            "sent_cap_vectors": sent_cap_vectors,
            "sent_cap_mask": sent_cap_mask,
            "sent_cap_len": sent_cap_len,
            "vid_id": get_data("vid_id"),
            "data_words": get_data("data_words")
        }


def create_datasets(
        dataset_path_dict, cfg, preload_vid_feat: bool,
        preload_text_feat: bool, eval=False):
    
    if eval:
        val_set = VideoDatasetFeatures(dataset_path_dict,
        cfg.dataset.val_split, cfg.dataset.max_frames, False, preload_vid_feat,
        preload_text_feat, 0)
        return val_set
    train_set = VideoDatasetFeatures(dataset_path_dict,
        cfg.CONFIG.COOT_DATA.TRAIN_SPLIT, cfg.CONFIG.COOT_DATA.MAX_FRAMES, True,
        preload_vid_feat, preload_text_feat, False)
    val_set = VideoDatasetFeatures(dataset_path_dict,
        cfg.CONFIG.COOT_DATA.VALIDATION_SPLIT, cfg.CONFIG.COOT_DATA.MAX_FRAMES, False, preload_vid_feat,
        preload_text_feat, 0)
    return train_set, val_set


def create_loaders(
        train_set: VideoDatasetFeatures, val_set: VideoDatasetFeatures,
        batch_size: int, num_workers: int, eval=False):
    if eval:
        val_loader = data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=val_set.collate_fn,
        pin_memory=True)
        return val_loader

    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=train_set.collate_fn,
        pin_memory=True)
    val_loader = data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=val_set.collate_fn,
        pin_memory=True)
    return train_loader, val_loader
