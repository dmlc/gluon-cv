python validate.py \
    --model alpha_pose_resnet101_v1b --dataset coco --num-joints 17 \
    --batch-size 128 --num-gpus 4 -j 60 \
    --params-file duc_se_coco.params \
    --input-size 320,256 --flip-test
