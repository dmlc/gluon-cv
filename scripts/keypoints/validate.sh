python validate.py \
    --model simple_pose_resnet50_v1b --num-joints 17 \
    --batch-size 128 --num-gpus 8 -j 60 --dtype float32 \
    --params-file params_simple_pose_resnet50_v1b_test_cudnn/simple_pose_resnet50_v1b-139.params