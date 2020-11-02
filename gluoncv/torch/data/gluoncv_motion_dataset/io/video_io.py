import os
import glob
import subprocess
import tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


class ColorSpace(object):
    RGB = 0
    BGR = 3
    GRAY = 2


convert_from_to_dict = {ColorSpace.BGR: {ColorSpace.RGB: cv2.COLOR_BGR2RGB,
                                         ColorSpace.GRAY: cv2.COLOR_BGR2GRAY},
                        ColorSpace.RGB: {ColorSpace.BGR: cv2.COLOR_RGB2BGR,
                                         ColorSpace.GRAY: cv2.COLOR_RGB2GRAY},
                        ColorSpace.GRAY: {ColorSpace.BGR: cv2.COLOR_GRAY2BGR,
                                          ColorSpace.RGB: cv2.COLOR_GRAY2RGB}}

FFMPEG_FOURCC = {
    'libx264': 0x21,
    'avc1': cv2.VideoWriter_fourcc(*'avc1'),
    'mjpeg': 0x6c,
    'mpeg-4': 0x20
}


def convert_color_from_to(frame, cs_from, cs_to):
    if cs_from not in convert_from_to_dict or cs_to not in convert_from_to_dict[cs_from]:
        raise Exception('color conversion is not supported')
    convert_spec = convert_from_to_dict[cs_from][cs_to]
    return cv2.cvtColor(frame, convert_spec)


def read_vid_rgb(file):
    cap = cv2.VideoCapture(file)
    all_ts = []
    all_frames = []
    while True:
        ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        frame = read_frame(cap)
        if frame is None:
            break
        all_frames.append(frame)
        all_ts.append(ts)

    fps = cap.get(cv2.CAP_PROP_FPS)
    return InMemoryVideo(all_frames, fps, frame_ts=all_ts)


def format_frame(frame, color_space=ColorSpace.RGB):
    if color_space != ColorSpace.BGR:
        frame = convert_color_from_to(frame, ColorSpace.BGR, color_space)
    return frame


def read_frame(cap):
    _, frame = cap.read()
    if frame is None:
        return frame
    return Image.fromarray(format_frame(frame, ColorSpace.RGB), 'RGB')


def read_img(file):
    frame = cv2.imread(file)
    if frame is None:
        return frame
    return Image.fromarray(format_frame(frame, ColorSpace.RGB), 'RGB')


def write_img(file, img, color_space=ColorSpace.RGB):
    img = convert_color_from_to(img, color_space, ColorSpace.BGR)
    cv2.imwrite(file, img)


class VideoBaseClass(object):
    def __init__(self):
        raise NotImplementedError()

    def __del__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def _set_frame_ndx(self, frame_num):
        raise NotImplementedError()

    def get_next_frame_time_stamp(self):
        raise NotImplementedError()

    def read(self):
        raise NotImplementedError()

    def __iter__(self):
        self._set_frame_ndx(0)
        return self

    def iter_frame_ts(self, start_ts=0):
        return FrameTimeStampIterator(self, start_ts)

    def next(self):
        return self.__next__()

    def __next__(self):
        ts = self.get_next_frame_time_stamp()
        frame = self.read()
        if frame is None:
            raise StopIteration()
        return frame, ts

    def __getitem__(self, frame_num):
        if self._next_frame_to_read != frame_num:
            self._set_frame_ndx(frame_num)
        ts = self.get_next_frame_time_stamp()
        return self.read(), ts

    @property
    def verified_len(self):
        return len(self)

    @property
    def fps(self):
        return self.get_frame_rate()

    @property
    def width(self):
        return self.get_width()

    @property
    def height(self):
        return self.get_height()

    def get_frame_ind_for_time(self, time_stamp):
        """
        Returns the index for the frame at the timestamp provided.
        The frame index returned is the first frame that occurs before or at the timestamp given.

        Args:
            time_stamp (int): the millisecond time stamp for the desired frame

        Returns (int):
            the index for the frame at the given timestamp.

        """
        assert isinstance(time_stamp, int)
        return int(self.fps * time_stamp / 1000.)

    def get_frame_for_time(self, time_stamp):
        return self[self.get_frame_ind_for_time(time_stamp)]

    def get_frame_rate(self):
        raise NotImplementedError()

    def get_width(self):
        raise NotImplementedError()

    def get_height(self):
        raise NotImplementedError()

    @property
    def duration(self):
        raise NotImplementedError()

    def asnumpy_and_ts(self):
        out = []
        out_ts = []
        for frame, ts in self.iter_frame_ts():
            out.append(frame)
            out_ts.append(ts)
        return out, out_ts

    def asnumpy(self):
        out = []
        for frame in self:
            out.append(frame)
        return out

    def num_frames(self):
        return len(self)

    def get_frame(self, index):
        return self[index]

    def get_frame_batch(self, index_list):
        '''
        Return a list of PIL Image classes
        Args:
            index_list (List[int]): list of indexes
            color_mode (str):  color mode of the pil image typically 'RGB'

        Returns: List[PIL.Image]

        '''
        return [self.get_frame(i) for i in index_list]


class FrameTimeStampIterator(object):
    def __init__(self, frame_reader, start_ts=0):
        self.frame_reader = frame_reader
        self.frame_reader._set_frame_time(start_ts)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        return next(self.frame_reader)


class InMemoryVideo(VideoBaseClass):
    def __init__(self, frames=None, fps=None, frame_ts=None):
        self._frames = []
        if frames is not None:
            self._frames = list(frames)

        self._fps = fps
        self._next_frame_to_read = 0

        self._frame_ts = []
        if len(self._frames) > 0:
            assert len(frame_ts) == len(self._frames)
            assert all(a <= b for a, b in zip(frame_ts[:-1], frame_ts[1:]))
            self._frame_ts = frame_ts

    def __del__(self):
        pass

    def __len__(self):
        return len(self._frames)

    def _set_frame_ndx(self, frame_num):
        self._next_frame_to_read = frame_num

    def get_next_frame_time_stamp(self):
        if self._next_frame_to_read >= len(self._frame_ts):
            return None
        return self._frame_ts[self._next_frame_to_read]

    def read(self):
        if self._next_frame_to_read >= len(self._frames):
            return None
        f = self._frames[self._next_frame_to_read]
        self._next_frame_to_read += 1
        return f

    def __setitem__(self, key, value):
        self._next_frame_to_read = key + 1
        self._frames[key] = value

    def append(self, frame, ts=None):
        assert ts is None or len(self._frame_ts) == 0 or ts > self._frame_ts[-1]
        self._frames.append(frame)
        self._next_frame_to_read = len(self._frames)
        if ts is None:
            if len(self._frame_ts) > 0:
                self._frame_ts.append(self._frame_ts[-1] + 1000. / self.fps)
            else:
                self._frame_ts.append(0.)
        else:
            self._frame_ts.append(ts)

    def extend(self, frames, tss):
        assert all(a <= b for a, b in zip(tss[:-1], tss[1:]))
        self._frames.extend(frames)
        self._frame_ts.extend(tss)
        self._next_frame_to_read = len(self._frames)

    def get_frame_rate(self):
        return self._fps

    def asnumpy(self):
        return self._frames

    def get_frame_ind_for_time(self, time_stamp):
        ind = np.searchsorted(self._frame_ts, time_stamp)
        if ind > 0:
            ind -= 1
        return ind


class InMemoryMXVideo(InMemoryVideo):
    def asnumpy(self):
        return [f.asnumpy() for f in self._frames]


img_exts = ['.jpg', '.jpeg', '.jp', '.png']
vid_exts = ['.avi', '.mpeg', '.mp4', '.mov']


class VideoFrameReader(VideoBaseClass):
    def __init__(self, file):
        self.cap = None
        self.file_name = file
        self._next_frame_to_read = 0
        self._verified_len = None
        self.frame_cache = {}
        self._is_vid = None
        self._is_img = None
        self._len = None
        self._duration = None

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

    @property
    def is_video(self):
        return not self.is_img

    @property
    def is_img(self):
        if self._is_img is None:
            _, ext = os.path.splitext(self.file_name)
            self._is_img = ext.lower() in img_exts
        return self._is_img

    def _lazy_init(self):
        if self.is_video and self.cap is None:
            self.cap = cv2.VideoCapture(self.file_name)

    def read_from_mem_cache(self):
        return None

    def read(self):
        self._lazy_init()
        if (not self.is_img) and self._next_frame_to_read != self.cap.get(cv2.CAP_PROP_POS_FRAMES):
            raise Exception("failed read frame check, stored {} , cap val {} , file {}".format(
                self._next_frame_to_read, self.cap.get(cv2.CAP_PROP_POS_FRAMES), self.file_name))
        if self.is_video:
            frame = read_frame(self.cap)
        else:
            if self._next_frame_to_read == 0:
                frame = read_img(self.file_name)
            else:
                frame = None
        if frame is None:
            self._verified_len = self._next_frame_to_read
        self._next_frame_to_read += 1
        return frame

    def _set_frame_ndx(self, frame_num):
        self._lazy_init()
        if self.is_video:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self._next_frame_to_read = frame_num

    def _set_frame_time(self, frame_ts):
        self._lazy_init()
        self.cap.set(cv2.CAP_PROP_POS_MSEC, frame_ts)
        self._next_frame_to_read = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

    def get_frame_for_time(self, time_stamp):
        self._lazy_init()
        if self.is_video:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, time_stamp)
            self._next_frame_to_read = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return self.read()

    def get_next_frame_time_stamp(self):
        self._lazy_init()
        if self.is_video:
            return max(0, int(self.cap.get(cv2.CAP_PROP_POS_MSEC)))
        else:
            return 0

    def _init_len_and_duration(self):
        if self._duration is None:
            self._lazy_init()
            pos = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            self._duration = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            self._len = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_MSEC, pos)

    def __len__(self):
        if self.is_video:
            self._init_len_and_duration()
            return self._len
        else:
            return 1

    @property
    def duration(self):
        self._init_len_and_duration()
        return self._duration

    @property
    def verified_len(self):
        if self.is_video:
            return self._verified_len
        else:
            return 1

    def get_frame_rate(self):
        self._lazy_init()
        if self.is_video:
            return self.cap.get(cv2.CAP_PROP_FPS)
        else:
            return 1

    def get_width(self):
        self._lazy_init()
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def get_height(self):
        self._lazy_init()
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


class VideoSortedFolderReader(VideoBaseClass):
    def __init__(self, data_path, fps, glob_pattern="*"):
        self._data_path = data_path

        self._glob_pattern = glob_pattern
        frame_paths = glob.glob(os.path.join(data_path, glob_pattern))
        self._frame_paths = sorted(frame_paths)

        self._next_frame_to_read = 0
        self._last_read_frame = None
        self._fps = fps
        self._period = 1.0 / fps * 1000

    def __del__(self):
        pass

    def __len__(self):
        return len(self._frame_paths)

    @property
    def duration(self):
        return round(self._period * len(self))

    def get_frame_rate(self):
        return self._fps

    def _set_frame_ndx(self, frame_num):
        self._next_frame_to_read = frame_num

    def _set_frame_time(self, frame_ts):
        self._set_frame_ndx(round(frame_ts / self._period))

    def get_next_frame_time_stamp(self):
        return int(self._next_frame_to_read * self._period)

    def read(self):
        read_idx = self._next_frame_to_read
        if read_idx >= len(self._frame_paths):
            return None
        frame = read_img(self._frame_paths[read_idx])
        self._last_read_frame = read_idx
        self._next_frame_to_read += 1
        return frame

    def get_image_ext(self):
        return Path(self._frame_paths[0]).suffix

    def get_frame_path(self, frame_num=None):
        if frame_num is None:
            frame_num = self._last_read_frame
        return self._frame_paths[frame_num]


def write_video_rgb(file, frames, fps=None):
    # check if data has the fps property (eg: InMemoryVideo, VideoFrameReader or VideoCacheReader)
    if fps is None:
        fps = 30
    try:
        fps = frames.fps
    except:
        pass

    # write the video data frame-by-frame
    writer = None
    for frame in frames:
        frame = np.asarray(frame)
        if writer is None:
            writer = cv2.VideoWriter(file, FFMPEG_FOURCC['libx264'],
                                     fps=fps, frameSize=frame.shape[1::-1],
                                     isColor=True)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    if writer is not None:
        writer.release()


def cmd_with_addn_args(cmd, addn_args):
    if addn_args:
        if isinstance(addn_args, str):
            addn_args = addn_args.split()
        return cmd + addn_args
    else:
        return cmd


def resize_and_write_video_ffmpeg(in_path, out_path, short_edge_res,
                                  scaling_algorithm="lanczos", raw_scale_input=None,
                                  keep_audio=True, addn_args=None):
    # See https://trac.ffmpeg.org/wiki/Scaling for scaling options / details

    if short_edge_res is not None and raw_scale_input is not None:
        raise ValueError("Either short_edge_res or raw_scale_input should be provided, not both")

    if short_edge_res is not None:
        # The input height, divided by the minimum of the width and height 
        # (so either = 1 or > 1) times the new short edge, 
        # then round to the nearest 2. 
        # We keep the aspect ratio of the width and make sure it is also divisible
        # by 2 by using '-2' (see the ffpeg scaling wiki)
        scale_arg = "-2:'round( ih/min(iw,ih) * {} /2)*2'".format(short_edge_res)

        # Alternatively:
        # scale_arg = "{res}:{res}:force_original_aspect_ratio=increase".format(res=short_edge_res)
        # In case the output has a non even dimension (e.g. 301) after rescaling, 
        # we crop the single extra pixel
        # crop_arg = "floor(iw/2)*2:floor(ih/2)*2"
    else:
        scale_arg = raw_scale_input

    if keep_audio:
        audio_arg = None
    else:
        audio_arg = "-an"

    scale_arg += ":flags={}".format(scaling_algorithm)

    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_file_path = os.path.join(tmp_path, os.path.basename(out_path))

        ffmpeg_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                      "-i", in_path, "-vf", "scale={}".format(scale_arg),
                      audio_arg]
        ffmpeg_cmd = cmd_with_addn_args(ffmpeg_cmd, addn_args)
        ffmpeg_cmd += ["-strict", "experimental", tmp_file_path]
        ffmpeg_cmd = [x for x in ffmpeg_cmd if x is not None]

        subprocess.run(ffmpeg_cmd, check=True)

        subprocess.run(["mv", tmp_file_path, out_path], check=True)


def resize_and_write_video(file, frames, short_edge_res, fps=None):
    # check if data has the fps property (eg: InMemoryVideo, VideoFrameReader or VideoCacheReader)
    if fps is None:
        fps = 30
    try:
        fps = frames.fps
    except:
        pass

    # write the video data frame-by-frame
    writer = None
    new_size = None
    for frame in frames:
        if new_size is None:
            factor = float(short_edge_res) / min(frame.size)
            new_size = [int(i * factor) for i in frame.size]

        frame_np = frame.resize(new_size)
        frame_np = np.asarray(frame_np)

        if writer is None:
            writer = cv2.VideoWriter(file, FFMPEG_FOURCC['libx264'],
                                     fps=fps, frameSize=frame_np.shape[1::-1],
                                     isColor=True)
        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        writer.write(frame_np)

    if writer is not None:
        writer.release()


def write_img_files_to_vid(out_file, in_files, fps=None):
    # check if data has the fps property (eg: InMemoryVideo, VideoFrameReader or VideoCacheReader)
    if fps is None:
        fps = 30

    # write the video data frame-by-frame
    writer = None
    for in_file in in_files:
        with open(in_file, 'rb') as fp:
            frame = Image.open(fp)
            frame = np.asarray(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if writer is None:
                writer = cv2.VideoWriter(out_file, FFMPEG_FOURCC['avc1'],
                                         fps=fps, frameSize=frame.shape[1::-1],
                                         isColor=True)
            writer.write(frame)

    if writer is not None:
        writer.release()


def write_img_files_to_vid_ffmpeg(out_file, in_files, fps=None):
    if fps is None:
        fps = 30
    input_str = "'\nfile '".join(in_files)
    input_str = "file '" + input_str + "'\n"

    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_file_path = os.path.join(tmp_path, os.path.basename(out_file))
        # See https://trac.ffmpeg.org/wiki/Slideshow 
        # for why we are using input_str like this (for concat filter)
        # Need -safe 0 due to:
        # https://stackoverflow.com/questions/38996925/ffmpeg-concat-unsafe-file-name
        ret = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                              "-f", "concat", "-safe", "0", "-r", str(fps), "-i", "/dev/stdin",
                              tmp_file_path],
                             input=input_str.encode('utf-8'), check=True)
        subprocess.run(["mv", tmp_file_path, out_file], check=True)
    return ret


def convert_vid_ffmpeg(in_path, out_path, addn_args=None):
    # muxing queue size bug workaround:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", in_path, "-max_muxing_queue_size", "99999"]
    cmd = cmd_with_addn_args(cmd, addn_args)
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_file_path = os.path.join(tmp_path, os.path.basename(out_path))
        cmd += [tmp_file_path]
        ret = subprocess.run(cmd, check=True)
        subprocess.run(["mv", tmp_file_path, out_path], check=True)
    return ret
