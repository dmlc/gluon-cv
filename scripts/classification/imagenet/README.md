# Image Classification

Here we present a number of examples to train gluon on image classification tasks.

## ImageNet

Here we present examples of training resnet/mobilenet on ImageNet dataset.

The main training script is `train_imagenet.py`. The script takes various parameters, thus we offer suggested parameters, and corresponding results.

### ResNet50_v2

Training a ResNet50_v2 can be done with:

```
python train_imagenet.py --batch-size 64 --num-gpus 4 -j 32 --mode hybrid\
    --num-epochs 120 --lr 0.1 --momentum 0.9 --wd 0.0001\
    --lr-decay 0.1 --lr-decay-epoch 30,60,90 --model resnet50_v2 
```

Results:

| Model        | Top-1 Error | Top-5 Error |
|--------------|-------------|-------------|
| ResNet50_v2  | 0.2428      | 0.0738      |

### MobileNetV2_1.0

Training a MobileNetV2_1.0 can be done with:

```
python train_imagenet.py --batch-size 64 --num-gpus 4 -j 32 --mode hybrid\
    --num-epochs 200 --lr 0.1 --wd 0.00004\
    --lr-decay 0.98 --lr-decay-period 1 --model mobilenetv2_1.0
```

Results:

| Model            | Top-1 Error | Top-5 Error |
|------------------|-------------|-------------|
| MobileNetV2_1.0  | 0.3008      |             |

### Finetuning

Finetune on `MINC-2500` with:

```
python3 finetune_minc.py --data ~/data/minc/minc-2500 --model resnet50_v2\
    -j 32 --num-gpus 4 --epochs 40 -b 64 --lr 0.001 --lr-factor 0.75 --lr-steps 10,20,30
```

