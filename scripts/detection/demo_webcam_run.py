import argparse
import time

import cv2
import gluoncv as gcv
import matplotlib.pyplot as plt
import mxnet as mx



parser = argparse.ArgumentParser(description="Webcam object detection script",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--num-frames', type=int, default=200,
                    help='number of frames to run the demo for. -1 means infinite')


args = parser.parse_args()

# Load the model
net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)

# Load the webcam handler
cap = cv2.VideoCapture(0)
time.sleep(1)  ### letting the camera autofocus

axes = None
NUM_FRAMES = args.num_frames
i = 0
while i < NUM_FRAMES or NUM_FRAMES == -1:
    i += 1

    # Load frame from the camera
    ret, frame = cap.read()

    # Image pre-processing
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=700)

    # Run frame through network
    class_IDs, scores, bounding_boxes = net(rgb_nd)

    # Display the result
    plt.cla()
    axes = gcv.utils.viz.plot_bbox(frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes, ax=axes)
    plt.draw()
    plt.pause(0.001)


cap.release()