python train.py \
    --model simple_pose_resnet50_v1d --mode hybrid --num-joints 17 \
    --lr 0.001 --lr-mode step --lr-decay-epoch 90,120 \
    --num-epochs 140 --batch-size 128 --num-gpus 8 -j 16 \
    --dtype float32 --no-wd \
    --save-dir params_simple_pose_resnet50_v1d \
    --logging-file simple_pose_resnet50_v1d.log
