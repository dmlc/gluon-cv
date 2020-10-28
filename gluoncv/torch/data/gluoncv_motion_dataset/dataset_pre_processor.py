import os
import random
import logging
import ray
from pathlib import Path
from tqdm import tqdm

from .io.video_io import VideoFrameReader, write_img_files_to_vid_ffmpeg, \
    resize_and_write_video_ffmpeg, convert_vid_ffmpeg
from .utils.serialization_utils import save_json
from .dataset import GluonCVMotionDataset, DataSample, FieldNames, get_resized_video_location, \
    get_vis_gt_location, get_vis_thumb_location, get_vis_video_location


def generate_resized_video(data_sample:DataSample, short_edge_res:int, overwrite=False, cache_dir=None,
                           upscale=True, use_vis_path=True, force_encode=False, encode_kwargs=None, **kwargs):
    if cache_dir is None:
        resized_path = get_resized_video_location(data_sample,short_edge_res)
    else:
        resized_path = data_sample.get_cache_file(cache_dir, extension='.mp4')
    if os.path.isfile(resized_path) and not overwrite:
        return
    os.makedirs(os.path.dirname(resized_path), exist_ok=True)

    orig_vid_path = get_vis_video_location(data_sample) if use_vis_path else data_sample.data_path
    if not upscale and min(data_sample.width, data_sample.height) <= short_edge_res:
        if force_encode:
            if encode_kwargs is None:
                encode_kwargs = {k: v for k, v in kwargs.items() if k == "addn_args"}
            convert_vid_ffmpeg(orig_vid_path, resized_path, **encode_kwargs)
        else:
            os.symlink(orig_vid_path, resized_path)
    else:
        resize_and_write_video_ffmpeg(orig_vid_path, resized_path, short_edge_res, **kwargs)

    return resized_path


def generate_video(data_sample:DataSample, force_encode=False, overwrite=False, **kwargs):
    new_file = get_vis_video_location(data_sample)
    if (os.path.isfile(new_file) and not overwrite) and not (force_encode and os.path.islink(new_file)):
        return
    os.makedirs(os.path.dirname(new_file), exist_ok=True)

    video_file = data_sample.data_path
    #### Generate Video ####
    if os.path.isdir(video_file):
        # the data is a set of images, generate a video
        img_files = [os.path.join(video_file, f) for f in sorted(os.listdir(video_file))]
        write_img_files_to_vid_ffmpeg(out_file=new_file, in_files=img_files, fps=data_sample.metadata['fps'])
    else:
        if not video_file.endswith(".mp4") or force_encode:
            # Convert the video to mp4
            convert_vid_ffmpeg(video_file, new_file, **kwargs)
        else:
            # the data is a video, symlink to it
            if os.path.exists(new_file):
                os.remove(new_file)
            os.symlink(video_file, new_file)

    return new_file


def generate_thumbnail(data_sample:DataSample, overwrite=False):
    video_thumbnail_frame = get_vis_thumb_location(data_sample)
    if os.path.isfile(video_thumbnail_frame) and not overwrite:
        return
    os.makedirs(os.path.dirname(video_thumbnail_frame), exist_ok=True)

    video_file = get_vis_video_location(data_sample)

    #### Generate Thumbnail ####
    vid = VideoFrameReader(video_file)
    img, ts = vid.get_frame(30) if len(vid) > 30 else vid.get_frame(0)
    img.thumbnail((300, 300))
    img.save(video_thumbnail_frame)

    return video_thumbnail_frame


def generate_gt_vis_json(data_sample:DataSample, cache_suffix="", overwrite=False):
    gt_file = get_vis_gt_location(data_sample, cache_suffix)
    if os.path.isfile(gt_file) and not overwrite:
        return
    #### Generate GT Track json ####
    os.makedirs(os.path.dirname(gt_file), exist_ok=True)

    vis_video_file = get_vis_video_location(data_sample)
    vis_vid = VideoFrameReader(vis_video_file)
    sample_dict = data_sample.to_dict(include_id=True)
    sample_dict[FieldNames.METADATA][FieldNames.FPS] = vis_vid.fps
    save_json(sample_dict, gt_file, indent=0)

    return gt_file


@ray.remote
def generate_files_for_one_sample_ray(data_sample:DataSample, generator_list, overwrite):
    generate_files_for_one_sample(data_sample, generator_list, overwrite)

@ray.remote
def generate_files_for_multi_samples_ray(data_samples, generator_list, overwrite):
    for id, data_sample in data_samples:
        generate_files_for_one_sample(data_sample, generator_list, overwrite)
        logging.info('Finished: {}'.format(data_sample.data_path))

def generate_files_for_one_sample(data_sample:DataSample, generator_list, overwrite):
    if os.path.isabs(data_sample.data_relative_path):
        logging.error("Relative path of sample id: {} is absolute and so we cannot add to the cache, skipping."
                      " Path: {}".format(data_sample.id, data_sample.data_relative_path))
        return
    for gen in generator_list:
        try:
            gen(data_sample, overwrite)
        except Exception as e:
            try:
                gen(data_sample, True)
            except Exception as e:
                logging.exception('Failed: {}'.format(data_sample.data_path))
                return


def generate_preprocess_files(part=0, parts=1,
                              annotation_file='./kinetics/annotation/anno_400.json',
                              use_ray=False, num_cpus=4, overwrite=False, force_encode=False, short_edge_res=256,
                              distributed=False, shuffle_seed=None, dataset=None):

    if dataset is None:
        dataset = GluonCVMotionDataset(annotation_file)

    generator_list = [
        lambda sample, overwrite: generate_video(sample, force_encode, overwrite),
        generate_thumbnail,
        lambda sample, overwrite: generate_gt_vis_json(sample, dataset.get_anno_subpath(), overwrite),
    ]

    samples = sorted(dataset.samples)
    if shuffle_seed is not None and shuffle_seed != "None":
        random.seed(shuffle_seed)
        random.shuffle(samples)
    samples = samples[part::parts]

    logging.info("Using ray {} ,  distributed: {}".format(use_ray, distributed))
    if use_ray:
        num_ray_threads = 500
        if distributed:
            ray.init(redis_address="localhost:6379")
        else:
            ray.init(num_cpus=num_cpus)
        ray.get([generate_files_for_multi_samples_ray.remote(samples[i::num_ray_threads], generator_list, overwrite) for i in range(num_ray_threads)])
    else:
        for id, data_sample in tqdm(samples, mininterval=1.0):
            generate_files_for_one_sample(data_sample, generator_list, overwrite)


if __name__ == '__main__':
    import fire
    fire.Fire(generate_preprocess_files)
