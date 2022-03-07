#!/bin/bash
date
echo "Args: $@"
env
echo "jobId: $AWS_BATCH_JOB_ID"
echo "jobQueue: $AWS_BATCH_JQ_NAME"
echo "computeEnvironment: $AWS_BATCH_CE_NAME"

SOURCE_REF=$1
WORK_DIR=$2
COMMAND=$3
SAVED_OUTPUT=$4
SAVE_PATH=$5
REMOTE=$6
DEVICE=${7:-gpu}

if [ ! -z $REMOTE ]; then
    git remote set-url origin $REMOTE
fi;

git fetch origin $SOURCE_REF:working
git checkout working
if [ $DEVICE == "cpu" ]; then
	python3 -m pip install -U --quiet "mxnet==1.7.0.post1"
    python3 -m pip install -U --quiet torch==1.6.0+cpu torchvision==0.7.0+cpu
else
	python3 -m pip install -U --quiet "mxnet-cu102==1.7.0"
    python3 -m pip install -U --quiet torch==1.6.0 torchvision==0.7.0
fi;

python3 -m pip install --quiet -e .
python3 -m pip install --quiet timm==0.5.4

cd $WORK_DIR
/bin/bash -o pipefail -c "$COMMAND"
COMMAND_EXIT_CODE=$?
if [[ -f $SAVED_OUTPUT ]]; then
  aws s3 cp $SAVED_OUTPUT s3://gluon-cv-dev/batch/$AWS_BATCH_JOB_ID/$SAVE_PATH;
elif [[ -d $SAVED_OUTPUT ]]; then
  aws s3 cp --recursive $SAVED_OUTPUT s3://gluon-cv-dev/batch/$AWS_BATCH_JOB_ID/$SAVE_PATH;
fi;
exit $COMMAND_EXIT_CODE
