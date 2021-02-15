"""02. Train Image Classification with Auto Estimator
=====================================================

This tutorial goes through the basic steps of using GluonCV auto estimator to train an image classifier
with custom hyper-parameters.
"""

##########################################################
# Train with default configurations
# ---------------------------------
from gluoncv.auto.estimators import ImageClassificationEstimator

##########################################################
# In this tutorial, we use a small sample dataset
train, _, test = ImageClassificationEstimator.Dataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
train, val, _ = train.random_split(val_size=0.1, test_size=0)

##########################################################
# Create an estimator with default configurations.
# We only change the number of GPUs to reflect the hardware constraint.
#
# Note that you may still launch training if no gpu is available(with {'gpus': []})
# but be prepared that this is painfully slow and not even possible to finish
# in case the dataset isn't tiny.
#
# We recommend that you use at least one nvidia gpu with more than 6GB free GPU memory.
classifier = ImageClassificationEstimator(
    {'gpus': [0], 'train': {'batch_size': 16, 'epochs': 2}})


##########################################################
# run fit on train/validation data
classifier.fit(train, val)

##########################################################
# Evaluate the final model on test set
eval_result = classifier.evaluate(test)
print("Top1/Top5 on test dataset: {}".format(eval_result))

##########################################################
# save to/from disk to be used later
classifier.save('classifier.pkl')
classifier2 = ImageClassificationEstimator.load('classifier.pkl')

##########################################################
# run prediction on test images
pred = classifier2.predict(test.iloc[0]['image'])
print('GroundTruth:', test.iloc[0])
print('prediction:', pred)

##########################################################
# Customize your training configurations
# --------------------------------------
# You may modify configurations to customize your training,
# supported fields:
print(ImageClassificationEstimator._default_cfg)

##########################################################
# For example, we could change the learning rate and batch size
new_classifier = ImageClassificationEstimator({'gpus': [0], 'train': {'batch_size': 16, 'lr': 0.01}})

##########################################################
# A more natural format for modifying individual hyperparameter
# is to edit the yaml file saved automatically in `self._logdir`, here we just show an example
# of how to copy/edit and load the modified configuration file back to the estimator in python
#
# You may edit the yaml file directly with a text editor
import shutil
import os
config_name = 'config.yaml'
shutil.copyfile(os.path.join(classifier2._logdir, config_name), os.path.join('.', config_name))
cfg = open(config_name).read()
print(cfg)

##########################################################
# Let's modify the network from resnet50_v1 to resnet18_v1b
import fileinput
with fileinput.FileInput(config_name,
                         inplace = True, backup ='.bak') as f:
    for line in f:
        if 'resnet50_v1' in line:
            new_line = line.replace('resnet50_v1', 'resnet18_v1b')
            print(new_line, end='')
        else:
            print(line, end='')

##########################################################
# The new classifier now should reflect the new configs we just edited
new_classifier2 = ImageClassificationEstimator(config_name)
