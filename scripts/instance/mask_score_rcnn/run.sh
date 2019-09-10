CUDNN_AUTOTUNE_DEFAULT=0 MXNET_GPU_MEM_POOL_TYPE=Round MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF=32 \
python3 train_mask_rcnn.py --gpus 0,1,2,3,4,5,6,7 --dataset coco --network \
                           resnet50_v1b --epochs 26 --lr-decay-epoch 17,23 --val-interval 2 --use-fpn -j 32\
                           --executor-threads 8
#--disable-hybridization

