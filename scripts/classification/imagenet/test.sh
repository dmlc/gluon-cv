#!/bin/bash

if [ -z "$MODEL" ]; then
  export MODEL=resnet18_v1
fi

if [ -z "$NUM_TRAINING_SAMPLES" ]; then
  export NUM_TRAINING_SAMPLES=1281167
fi

if [ -z "$NUM_EPOCHS" ]; then
  export NUM_EPOCHS=3
fi

if [ -z "$NUM_GPUS" ] || [ $NUM_GPUS '-lt' 0 ]; then
  export NUM_GPUS=0
fi

if [ -z "$DATA_BACKEND" ]; then
  export DATA_BACKEND='mxnet'  # Options are: dali-gpu, dali-cpu, mxnet
fi

if [ -z "$TRAIN_DATA_DIR" ]; then
  export TRAIN_DATA_DIR=~/.mxnet/datasets/imagenet
fi

if [ -z "$DALI_VER" ]; then
  export DALI_VER=nvidia-dali-cuda100
fi

python train_imagenet.py --model $MODEL --data-backend $DATA_BACKEND --num-gpus $NUM_GPUS \
      --num-epochs $NUM_EPOCHS --num-training-samples $NUM_TRAINING_SAMPLES --use-rec \
       --rec-train $TRAIN_DATA_DIR/train.rec --rec-train-idx $TRAIN_DATA_DIR/train.idx \
       --rec-val $TRAIN_DATA_DIR/val.rec --rec-val-idx $TRAIN_DATA_DIR/val.idx --data-dir $TRAIN_DATA_DIR \


