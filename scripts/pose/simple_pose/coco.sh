python train_simple_pose.py \
    --model simple_pose_resnet50_v1b --mode hybrid --num-joints 17 \
    --lr 0.001 --wd 0.0 --lr-mode step --lr-decay-epoch 90,120 \
    --num-epochs 140 --batch-size 32 --num-gpus 8 -j 60 \
    --dtype float32 --warmup-epochs 0 --use-pretrained-base \
    --save-dir params_simple_pose_resnet50_v1b \
    --logging-file simple_pose_resnet50_v1b.log --log-interval 100
