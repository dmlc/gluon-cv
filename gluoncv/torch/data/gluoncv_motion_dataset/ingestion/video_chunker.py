import logging
import os
import pickle
import tqdm
from pathlib import Path

from ..dataset import GluonCVMotionDataset, DataSample, FieldNames, SplitNames
from ..utils.ingestion_utils import crop_video, process_dataset_splits, crop_fn_ffmpeg, \
    crop_fn_frame_folder


_log = logging.getLogger()
_log.setLevel(logging.DEBUG)


def even_chunk(sample, chunk_mins, chunks_per_vid=None, center=False, min_chunk_ms=3000):
    """
    A function that when given a sample returns evenly spaced timestamp chunks
    that can be used to crop the video

    :param sample: 
        The datasample to be chunked
    :param chunk_mins: 
        The desired length of each chunk in minutes. 
        This will be the length of all returned chunks unless:
        1. The sample is shorter than chunk_mins OR 
        2. None is provided to chunks_per_vid, then the last chunk might be a shorter length.
    :param chunks_per_vid: 
        The desired number of chunks per video. 
        If a video is too short to be chunked this many times,
        then the number of chunks will be reduced such that they all fit fully. 
        If the video is shorter than a single,
        chunk a single timestamp of the entire video will be returned. 
        If None is provided (the default),
        the video will be chunked fully and the final chunk might be a shorter length
        (but not shorter than min_chunk_ms).
    :param center: 
        Whether to center the chunks (add space before the first and after the last chunk).
        False by default.
    :param min_chunk_ms: 
        Only used when chunks_per_vid is None (chunk the whole video), 
        if the final chunk is smaller than this it will be excluded
    :return: 
        A list of timestamp pairs where each pair is a tuple of (start_time, end_time) for the chunk
    """
    duration = sample.frame_reader.duration
    chunk_ms = int(chunk_mins * 60) * 1000
    if chunks_per_vid is None:
        num_chunks = duration // chunk_ms
        # The final chunk has to be at least a minimum length (or the whole video)
        if duration % chunk_ms >= min_chunk_ms or num_chunks == 0:
            num_chunks += 1
        remaining_ms = 0
    else:
        num_chunks = chunks_per_vid
        remaining_ms = duration - (chunk_ms * num_chunks)
    if remaining_ms < 0:
        # Reduce num chunks by overrun
        # Equivalent of ceil when int dividing with -ve numbers,
        num_chunks += remaining_ms // chunk_ms
        num_chunks = max(1, num_chunks)
        remaining_ms = max(0, duration - (chunk_ms * num_chunks))
    num_spaces = max(1, num_chunks - 1 + (center * 2))
    spacing_ms = remaining_ms // num_spaces
    init_space = spacing_ms if center else 0
    time_starts = [i * (chunk_ms + spacing_ms) + init_space for i in range(num_chunks)]
    time_pairs = [(x, min(x + chunk_ms, duration)) for x in time_starts]
    return time_pairs

def ratio_chunk(sample, ratios):
    """
    :param sample: 
        The datasample to be chunked
    :param ratios: 
        The ratios to be chunked, in order relative to the video.
        If negative it will throw away that chunk
    :return: 
        See :func:`even_chunk`
    """
    if sum(abs(x) for x in ratios) > 1 + 1e-2:
        raise ValueError("Ratios add up to more than 1")

    duration = sample.frame_reader.duration
    time_pairs = []
    start_time = 0
    for ratio in ratios:
        end_time = start_time + (abs(ratio) * duration)
        if ratio > 0:
            time_pair = (round(start_time), round(end_time))
            time_pairs.append(time_pair)
        start_time = end_time
    return time_pairs


def align_times_to_framerate(time_pairs, fps):
    period = 1000 / fps
    aligned_pairs = []
    for time_pair in time_pairs:
        aligned_pair = tuple(round(round(t / period) * period) for t in time_pair)
        aligned_pairs.append(aligned_pair)
    return aligned_pairs


def update_anno(sample: DataSample, new_sample: DataSample, time_pair: (int, int)):
    start_time, end_time = time_pair
    for ts, entities in sample.time_entity_dict.items():
        if ts >= start_time and ts < end_time:
            for entity in entities:
                new_entity = pickle.loads(pickle.dumps(entity))
                new_entity.time = entity.time - start_time
                # This assumes constant fps
                new_entity.frame_num = round((new_entity.time / 1000) * new_sample.fps)
                new_sample.add_entity(new_entity)
    return new_sample


def add_orig_samples(new_dataset, dataset):
    for sid, sample in dataset.get_split_samples("test"):
        new_dataset.add_sample(sample)


def write_new_split(dataset, time_pairs):
    def split_func(sample):
        time_pair = time_pairs.get(sample.id)
        if time_pair is None:
            return SplitNames.TEST
        start_time, end_time = time_pair
        if start_time == 0:
            return SplitNames.TRAIN
        else:
            return SplitNames.VAL

    process_dataset_splits(dataset, split_func, save=True)


def main(anno_path:str="./annotation/anno.json", new_name=None,
         chunk_mins=5, chunks_per_vid=1, chunk_func=even_chunk,
         link_unchanged=True, cache_name=None,input_cache=None,
         name_suffix="", fps=None, split=None, new_split=None,
         overwrite=False, part=0, parts=1):

    fps_suffix = "_{}r".format(fps) if fps is not None else ""
    if name_suffix:
        name_suffix = "_" + name_suffix
    if new_name is None:
        new_name = "anno_chunks_{}m_{}p{}{}.json".format(chunk_mins,
                                                         chunks_per_vid,
                                                         fps_suffix,
                                                         name_suffix)
    if cache_name is None:
        cache_name = "vids_chunked/vid_chunks_{}m_{}p{}{}".format(chunk_mins,
                                                                  chunks_per_vid,
                                                                  fps_suffix,
                                                                  name_suffix)

    dataset = GluonCVMotionDataset(anno_path)
    new_dataset = GluonCVMotionDataset(new_name,
                                       dataset.root_path,
                                       split_file=new_split,
                                       load_anno=False)

    chunk_cache_dir = Path(dataset.cache_root_path, cache_name)
    chunk_data_link = Path(dataset.data_root_path, cache_name)
    os.makedirs(chunk_cache_dir, exist_ok=True)
    os.makedirs(chunk_data_link.parent, exist_ok=True)
    try:
        os.symlink(chunk_cache_dir, chunk_data_link)
    except OSError:
        pass  # already exists

    new_sample_times = {}
    samples = dataset.get_split_samples(split)
    samples = samples[part::parts]
    for sample_id, sample in tqdm.tqdm(samples, mininterval=1):
        time_pairs = ratio_chunk(sample, [0.5, 0.5])
        time_pairs = align_times_to_framerate(time_pairs, sample.fps)
        out_suffix = ".mp4"
        if input_cache is not None:
            input_path = sample.get_cache_file(input_cache, out_suffix)
        else:
            input_path = sample.data_path
        output_path = chunk_data_link / sample.data_relative_path
        is_frame_dir = Path(input_path).is_dir()
        if not is_frame_dir:
            output_path = output_path.with_suffix(out_suffix)

        for start_time, end_time in time_pairs:
            if link_unchanged and len(time_pairs) == 1 \
                    and start_time == 0 and abs(sample.frame_reader.duration - end_time) < 50 \
                    and (fps is None or sample.frame_reader.fps == fps):
                if not output_path.exists():
                    os.makedirs(output_path.parent, exist_ok=True)
                    os.symlink(input_path, output_path)
                cropped = False
                full_output_path = output_path
            else:
                if is_frame_dir:
                    # fps for folder frame reader, not output fps
                    addn_args = dict(crop_fn=crop_fn_frame_folder, in_fps=sample.fps)
                else:
                    # output fps
                    addn_args = dict(crop_fn=crop_fn_ffmpeg, fps=fps)
                cropped, full_output_path = crop_video(input_path, str(output_path),
                                                       start_time, end_time,
                                                       overwrite=overwrite, add_output_ts=True,
                                                       make_dirs=True, **addn_args)
            new_data_path = Path(full_output_path).relative_to(dataset.data_root_path)
            new_id = f"{sample_id}_{start_time}-{end_time}"
            new_sample = sample.get_copy_without_entities(new_id=new_id)
            new_sample.data_relative_path = new_data_path
            fr = new_sample.frame_reader
            new_sample.metadata[FieldNames.NUM_FRAMES] = fr.num_frames()
            new_sample.metadata[FieldNames.DURATION] = fr.duration
            new_sample.metadata[FieldNames.ORIG_ID] = sample_id
            update_anno(sample, new_sample, (start_time, end_time))
            if cropped:
                _log.info("Done crop for {}".format(new_data_path))
            new_dataset.add_sample(new_sample)
            new_sample_times[new_sample.id] = (start_time, end_time)

    if parts == 1:
        add_orig_samples(new_dataset, dataset)
        new_dataset.dump()

    return True, new_dataset


if __name__ == "__main__":
    import fire
    fire.Fire(main)
