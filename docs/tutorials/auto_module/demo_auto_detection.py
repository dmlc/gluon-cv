"""03. Train classifier or detevctor with HPO using GluonCV Auto task
=====================================================================

The previous image classification example shows the basic usages to train/evaluate/predict
using estimators provided by `gluoncv.auto.estimators`. Similarly, you can train object detectors
using  `SSDEstimator`, `YOLOv3Estimator`, `CenterNetEstimator`, `FasterRCNNEstimator`.

In this tutorial, we will move forward a little bit, into the hyper-parameter tunning space!
We will show you how to offload an experiment to the backend HPO searcher, to offer better result if
computational cost is abundant.
"""

##########################################################
# Train with default configurations
# ---------------------------------
from gluoncv.auto.tasks.object_detection import ObjectDetection
import autogluon.core as ag

##########################################################
# In this tutorial, we use a small sample dataset
train = ObjectDetection.Dataset.from_voc(
    'https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip')
train, val, test = train.random_split(val_size=0.1, test_size=0.1)

##########################################################
# Define search space
#
# We use very conservative search space to reduce CI runtime,
# you may adjust the `num_trials`, `time_limits` and `epochs` in order
# to achieve better results
time_limits = 60 * 60  # 1hr
search_args = {'lr': ag.Categorical(1e-3, 1e-2),
               'num_trials': 2,
               'epochs': 2, 'num_workers': 16,
               'ngpus_per_trial': 1,
               'search_strategy': 'random',
               'time_limits': time_limits}

##########################################################
# Construct a object detection task based on the config.
task = ObjectDetection(search_args)

##########################################################
# Automatically fit a model.
detector = task.fit(train, val)

##########################################################
# Evaluate the final model on test set.
test_map = detector.evaluate(test)
print("mAP on test dataset: {}".format(test_map[-1][-1]))
print(test_map)

##########################################################
# Save our final model.
detector.save('detector.pkl')
