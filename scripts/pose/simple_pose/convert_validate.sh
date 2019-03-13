python validate.py \
    --model simple_pose_resnet50_v1b --num-joints 17 \
    --batch-size 128 --num-gpus 8 -j 60 --dtype float32 \
    --mean 0.406,0.456,0.485 --std 0.225,0.224,0.229 --score-threshold 0.2 \
    --params-file simple_pose_resnet50_v1b_converted.params
