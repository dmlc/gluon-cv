import glob
import numpy as np
import os
import re
import tempfile
from addict import Dict
from collections import defaultdict
from pathlib import Path
from subprocess import call
import cv2

from ..dataset import GluonCVMotionDataset, DataSample, FieldNames, get_resized_video_location
from ..io.video_io import VideoFrameReader, VideoSortedFolderReader, cmd_with_addn_args, \
    FFMPEG_FOURCC


def default_filepath_filter(path):
    return not path.name.startswith('.')


def get_filepaths(root_path, ext=".mp4", replace_path=None, filter_fn=default_filepath_filter,
                  glob_exp=None, parents_only=False):
    if replace_path is None:
        replace_path = root_path
    if glob_exp is None:
        glob_exp = '*{}'.format(ext)
    paths = glob.glob(str(Path(root_path) / "**" / glob_exp), recursive=True)
    paths = [Path(p) for p in paths]
    if replace_path:
        rel_paths = [p.relative_to(replace_path) for p in paths]
    else:
        rel_paths = paths
    if filter_fn:
        rel_paths = [p for p in rel_paths if filter_fn(p)]
    if parents_only:
        parents = {rel_path.parent for rel_path in rel_paths}
        rel_paths = list(parents)
    return rel_paths


def process_dataset_splits(dataset, sample_split_fn, save=False, log=True):
    splits = defaultdict(list)
    for sample_id, sample in dataset.samples:
        sample_split = sample_split_fn(sample)
        if sample_split is not None:
            splits[sample_split].append(sample_id)

    if log:
        print("Processed {} samples".format(len(dataset.samples)))
        for split_name, split_ids in splits.items():
            print("Found {} samples for {}".format(len(split_ids), split_name))

    if save:
        if log:
            print("Saving data splits to: {}".format(dataset.split_path))
        dataset.splits = splits
        dataset.dump_splits()

    return splits


def get_datasample_from_video(rel_path, full_path=None, dataset=None):
    if full_path is None:
        if dataset is not None:
            full_path = os.path.join(dataset.data_root_path, rel_path)
        else:
            raise ValueError("Both full path and dataset cannot be empty")
    vid_reader = VideoFrameReader(full_path)

    sample = DataSample(rel_path)

    fps = vid_reader.fps
    num_frames = len(vid_reader)
    duration = vid_reader.duration
    frame = vid_reader.read()
    if frame is None:
        return None
    width = frame.width
    height = frame.height

    metadata = {
        FieldNames.DATA_PATH: rel_path,
        FieldNames.FPS: fps,
        FieldNames.DURATION: duration,
        FieldNames.NUM_FRAMES: num_frames,
        FieldNames.RESOLUTION: {"width": width, "height": height},
    }
    sample.metadata = metadata

    return sample


def check_dimensions(dataset, target_sidelen=256, cache_func=None):
    if cache_func is None:
        cache_func = lambda sample: get_resized_video_location(sample, target_sidelen)
    bad_vids = []
    for i, (id, sample) in enumerate(dataset.samples):
        vid_path = cache_func(sample)
        if not os.path.exists(vid_path):
            print("Warning, missing cache vid path {} , skipping".format(vid_path))
            continue

        vid_reader = VideoFrameReader(vid_path)
        if target_sidelen not in (vid_reader.width, vid_reader.height):
            print("Warning, target side length {} not in video, actual res: {}x{} , \
                vid path: {}".format(
                target_sidelen, vid_reader.width, vid_reader.height, vid_path))
            bad_vids.append(id)

        if (i + 1) % 100 == 0:
            print("Done checking {} vids".format(i + 1))

    print("Total of {} bad sized vids".format(len(bad_vids)))
    print(bad_vids)
    return bad_vids


def crop_fn_ffmpeg(input_path, output_path, t1, t2, fps=None, addn_args=None):
    t1_secs, t2_secs = [t / 1000.0 for t in (t1, t2)]

    # You need -strict experimental on some versions of ffmpeg (including the ubuntu default) 
    # or it will throw an error
    # when decoding aac audio, see: https://stackoverflow.com/a/32932092 
    # and https://superuser.com/a/543593
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", "{:.3f}".format(t1_secs),
        "-i", input_path,
        "-t", "{:.3f}".format(t2_secs - t1_secs),
    ]
    if fps is not None:
        cmd += ["-r", str(fps)]
    cmd = cmd_with_addn_args(cmd, addn_args)
    cmd += ["-strict", "experimental", output_path]

    return call(cmd)


def crop_fn_videoframereader_opencv(input_path, output_path, t1, t2):
    start_time, end_time = t1, t2

    frame_reader = VideoFrameReader(input_path)
    fps = frame_reader.fps
    ts = start_time
    frame = frame_reader.get_frame_for_time(ts)

    # We are using the VideoWriter directly instead of write_video_rgb 
    # due to memory issues of holding all frames
    # in memory before writing
    writer = cv2.VideoWriter(output_path, FFMPEG_FOURCC['avc1'], fps=fps,
                             frameSize=(frame.width, frame.height), isColor=True)
    while ts <= end_time and frame is not None:
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

        ts = frame_reader.get_next_frame_time_stamp()
        frame = frame_reader.read()

    writer.release()

    return True


def crop_fn_frame_folder(input_path, output_path, t1, t2, in_fps=None,
                         frame_reader=None, symlink=True,
                         copy_names=True, copy_frames=True):
    if in_fps is None and frame_reader is None:
        raise ValueError("Both in_fps and frame_reader can't be None")

    if frame_reader is None:
        frame_reader = VideoSortedFolderReader(input_path, fps=in_fps)

    start_time, end_time = t1, t2

    ts = start_time
    image_ext = frame_reader.get_image_ext()

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    for frame_idx, (frame, ts) in enumerate(frame_reader.iter_frame_ts(ts)):
        if ts > end_time:
            break

        frame_path = frame_reader.get_frame_path()
        if copy_names:
            im_filename = Path(frame_path).name
        else:
            # 6 digits is about 9 hours of video at 30 fps
            im_filename = f"{frame_idx:06}" + image_ext
        im_out_path = Path(output_path, im_filename)
        if symlink:
            os.symlink(frame_path, im_out_path)
        elif copy_frames:
            call(["cp", frame_path, im_out_path])
        else:
            frame.save(im_out_path)

    return True


def crop_video(input_path, output_path, start_time, end_time,
               overwrite=False, min_video_length_ms=1000,
               crop_fn=crop_fn_ffmpeg, add_output_ts=False,
               make_dirs=False, *args, **kwargs):
    if add_output_ts:
        output_base, ext = os.path.splitext(output_path)
        output_path = "{name}_{t1}-{t2}{ext}".format(name=output_base,
                                                     t1=start_time,
                                                     t2=end_time,
                                                     ext=ext)

    if not os.path.exists(input_path):
        raise ValueError("input video path: {} does not exist".format(input_path))

    if end_time - start_time < min_video_length_ms:
        raise ValueError("End time - start time must be >= {min_len}, \
                          error for video: {path}".format(
                            min_len=min_video_length_ms, path=input_path))

    parent_dir = os.path.dirname(output_path)
    if not os.path.exists(parent_dir):
        if make_dirs:
            os.makedirs(parent_dir, exist_ok=True)
        else:
            raise ValueError("Parent output directory for {} \
                              does not exist for input video path: {}" .format(
                                output_path, input_path))

    if os.path.exists(output_path) and (not overwrite):
        return False, output_path

    with tempfile.TemporaryDirectory() as tmp_crop_dir_path:
        tmp_crop_path = os.path.join(tmp_crop_dir_path, os.path.basename(output_path))
        crop_fn(input_path, tmp_crop_path, start_time, end_time, *args, **kwargs)

        ret_code = call(['mv', tmp_crop_path, output_path])
        if ret_code:
            raise Exception('Failed to move to {} from tmp location'.format(output_path))

    return True, output_path


class AddictList(Dict):
    """Class that can be used for gathering statistics 
    about data by easily storing lists of different data"""

    LKEY = "lst"

    def _get_lst(self):
        lkey = self.LKEY
        if lkey not in self:
            self[lkey] = []
        return self[lkey]

    def append(self, item):
        self._get_lst().append(item)

    def extend(self, lst2):
        self._get_lst().extend(lst2)

    def replace_lst(self):
        for k, v in self.items():
            if isinstance(v, type(self)):
                if len(v) == 1 and self.LKEY in v:
                    self[k] = v[self.LKEY]
                else:
                    v.replace_lst()


chunk_pat = re.compile(r"_\d+-\d+$")
def default_chunked_sample_to_orig_id(chunked_sample, chunk_subdir="", orig_dataset=None):
    id_path = Path(chunked_sample.id)
    chunked_id_stem = id_path.stem
    orig_id = chunk_pat.sub('', chunked_id_stem, count=1)
    orig_id = str(id_path.with_name(orig_id + id_path.suffix).relative_to(chunk_subdir))
    return orig_id


def get_orig_chunked_sample(sample: DataSample, orig_dataset: GluonCVMotionDataset,
                            chunk_subdirs: [] = None,
                            sample_map_func=default_chunked_sample_to_orig_id,
                            exclude_unchunked=False):
    if chunk_subdirs:
        chunk_subdir = [subdir for subdir in chunk_subdirs \
                        if sample.data_relative_path.startswith(str(subdir))]
        if not chunk_subdir:
            return None
        chunk_subdir = str(chunk_subdir[0])
    else:
        chunk_subdir = Path(sample.data_relative_path).parts[0]
    orig_id = sample_map_func(sample, chunk_subdir, orig_dataset)
    if orig_id not in orig_dataset:
        return None
    orig_sample = orig_dataset[orig_id]
    if exclude_unchunked and \
            Path(orig_sample.data_relative_path) == \
            Path(sample.data_relative_path).relative_to(chunk_subdir):
        return None
    return orig_sample


def get_chunked_id_map(chunked_dataset, orig_dataset, *args, **kwargs):
    id_map = {}
    for chunked_id, sample in chunked_dataset.samples:
        orig_sample = get_orig_chunked_sample(sample, orig_dataset, *args, **kwargs)
        orig_id = orig_sample.id
        if orig_id is not None:
            id_map[chunked_id] = orig_id
    return id_map
