#!/usr/bin/env bash

COVER_PACKAGE=$1
TESTS_PATH=$2

EFS=/mnt/efs

mkdir -p ~/.mxnet/models
for f in $EFS/.mxnet/models/*.params; do
    ln -s $f ~/.mxnet/models/$(basename "$f")
done

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MPLBACKEND=Agg
export KMP_DUPLICATE_LIB_OK=TRUE

if [[ $TESTS_PATH == *"auto"* ]]; then
  echo "Installing autogluon.core and timm for auto module"
  pip3 install autogluon.core==0.2.0
  pip3 install timm==0.5.4
fi

nosetests --process-restartworker --with-timer --timer-ok 5 --timer-warning 20 -x --with-coverage --cover-package $COVER_PACKAGE -v $TESTS_PATH
