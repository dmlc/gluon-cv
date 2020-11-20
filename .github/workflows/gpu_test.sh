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
nosetests --with-timer --timer-ok 5 --timer-warning 20 -x --with-coverage --cover-package $COVER_PACKAGE -v $TESTS_PATH
