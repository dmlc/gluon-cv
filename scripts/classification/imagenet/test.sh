#!/bin/bash

if [ -z "$NETWORK" ]; then
  export NETWORK=resnet-v1
fi

if [ -z "$NUM_EXAMPLES" ]; then
  export NUM_EXAMPLES=102400
fi

if [ -z "$NUM_EPOCHS" ]; then
  export NUM_EPOCHS=1
fi

pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100

if [ -z "$NUM_GPUS" ] || [ $NUM_GPUS '-le' 0 ]; then
  python train_imagenet_runner --use_all_gpus --seed 42 -e $NUM_EPOCHS -b 128 -s $NUM_EXAMPLES --network $NETWORK $NO_DALI
else
  python train_imagenet_runner -n $NUM_GPUS --seed 42 -e $NUM_EPOCHS -b 128 -s $NUM_EXAMPLES --network $NETWORK $NO_DALI
fi

