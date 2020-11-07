"""10. Introducing Decord: an efficient video reader
====================================================

Training deep neural networks on videos is very time consuming. For example, training a state-of-the-art SlowFast network
on Kinetics400 dataset using a server with 8 V100 GPUs takes more than 10 days. Slow training causes long research cycles
and is not friendly for new comers and students to work on video related problems. There are several reasons causing the slowness,
big batch of data, inefficiency of video reader and huge model computation.

Another troubling matter is the complex data preprocessing and huge storage cost. Take Kinetics400 dataset as an example, this dataset
has about 240K training and 20K validation videos. All the videos take 450G disk space.
However, if we decode the videos to frames and use image loader to train the model, the decoded frames will take 6.8T disk space, which
is unacceptable to most people. In addition, the decoding process is slow. It takes 1.5 days using 60 workers to decode all the videos to frames.
If we use 8 workers (as in common laptop or standard workstation), it will take a week to perform such data preprocessing even before your actual training.

Given the challenges aforementioned, in this tutotial, we introduce a new video reader, `Decord <https://github.com/zhreshold/decord>`_.
Decord is efficient and flexible. It provides convenient video slicing methods based on a wrapper on top of hardware accelerated video decoders,
e.g. FFMPEG/LibAV and Nvidia Codecs. It is designed to handle awkward video shuffling experience in order to provide smooth experiences
similar to random image loader for deep learning. In addition, it works cross-platform, e.g., Linux, Windows and Mac OS.
With the new video reader, you don't need to decode videos to frames anymore, just start training on your video dataset with even higher training speed.

"""


########################################################################
# Install
# -------
#
# Decord is easy to install, just
# ::
#
#     pip install decord

########################################################################
# Usage
# -----
#
# We provide some usage cases here to get you started. For complete API, please refer to official documentation.

################################################################
# Suppose we want to read a video. Let's download the example video first.
from gluoncv import utils
url = 'https://github.com/bryanyzhu/tiny-ucf101/raw/master/abseiling_k400.mp4'
video_fname = utils.download(url)

from decord import VideoReader
vr = VideoReader(video_fname)

################################################################
# If we want to load the video in a specific dimension so that it can be fed into a CNN for processing,

vr = VideoReader(video_fname, width=320, height=256)

################################################################
# Now we have loaded the video, if we want to know how many frames are there in the video,

duration = len(vr)
print('The video contains %d frames' % duration)

################################################################
# If we want to access frame at index 10,

frame = vr[9]
print(frame.shape)

################################################################
# For deep learning, usually we want to get multiple frames at once. Now you can use ``get_batch`` function,
# Suppose we want to get a 32-frame video clip by skipping one frame in between,

frame_id_list = range(0, 64, 2)
frames = vr.get_batch(frame_id_list).asnumpy()
print(frames.shape)

################################################################
# There is another advanced functionality, you can get all the key frames as below,
key_indices = vr.get_key_indices()
key_frames = vr.get_batch(key_indices)
print(key_frames.shape)

################################################################
# Pretty flexible, right? Try it on your videos.

################################################################
# Speed comparison
# ----------------

################################################################
# Now we want to compare its speed with Opencv VideoCapture to demonstrate its efficiency.
# Let's load the same video and get all the frames randomly using both decoders to compare their performance.
# We will run the loading for 11 times: use the first one as warming up, and average the rest 10 runs as the average speed.

import cv2
import time
import numpy as np

frames_list = np.arange(duration)
np.random.shuffle(frames_list)

# Decord
for i in range(11):
    if i == 1:
        start_time = time.time()
    decord_vr = VideoReader(video_fname)
    frames = decord_vr.get_batch(frames_list)
end_time = time.time()
print('Decord takes %4.4f seconds.' % ((end_time - start_time)/10))

# OpenCV
for i in range(11):
    if i == 1:
        start_time = time.time()
    cv2_vr = cv2.VideoCapture(video_fname)
    for frame_idx in frames_list:
        cv2_vr.set(1, frame_idx)
        _, frame = cv2_vr.read()
    cv2_vr.release()
end_time = time.time()
print('OpenCV takes %4.4f seconds.' % ((end_time - start_time)/10))

################################################################
# We can see that Decord is 2x faster than OpenCV VideoCapture.
# We also compare with `Pyav container <https://github.com/mikeboers/PyAV>`_ and demonstrate 2x speed up as well.
#
# In conclusion, Decord is an efficient and flexible video reader. It supports get_batch, GPU loading, fast random access, etc, which is
# perfectly designed for training video deep neural networks. We use Decord in our video model training for large-scale datasets and observe
# similar speed as using image loaders on decoded video frames. This significanly reduces the data preprocessing time and the storage
# cost for large-scale video datasets.
