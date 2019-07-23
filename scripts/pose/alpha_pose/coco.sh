python train_alpha_pose.py --dataset coco \
    --model alpha_pose_resnet101_v1b --mode hybrid --num-joints 17 \
    --lr 0.001 --wd 0.0 --lr-mode step --lr-decay-epoch 90,120 \
    --num-epochs 140 --batch-size 32 --num-gpus 4 -j 60 \
    --dtype float32 --warmup-epochs 0 --use-pretrained-base \
    --save-dir params_alpha_pose_resnet101_v1b_coco \
    --logging-file alpha_pose_resnet101_v1b_coco.log --log-interval 100 --flip-test
