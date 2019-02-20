python validate.py \
    --model simple_pose_resnet18_v1b --num-joints 17 \
    --batch-size 128 --num-gpus 8 -j 60
python validate.py \
    --model simple_pose_resnet50_v1b --num-joints 17 \
    --batch-size 128 --num-gpus 8 -j 60
python validate.py \
    --model simple_pose_resnet50_v1d --num-joints 17 \
    --batch-size 128 --num-gpus 8 -j 60
python validate.py \
    --model simple_pose_resnet101_v1b --num-joints 17 \
    --batch-size 128 --num-gpus 8 -j 60
python validate.py \
    --model simple_pose_resnet101_v1d --num-joints 17 \
    --batch-size 128 --num-gpus 8 -j 60
python validate.py \
    --model simple_pose_resnet152_v1b --num-joints 17 \
    --batch-size 128 --num-gpus 8 -j 60
python validate.py \
    --model simple_pose_resnet152_v1d --num-joints 17 \
    --batch-size 128 --num-gpus 8 -j 60
