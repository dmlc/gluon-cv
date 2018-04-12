# Image Classification

Here we present a number of examples to train gluon on image classification tasks.

## CIFAR10 

Here we present examples of training resnet/wide-resnet on CIFAR10 dataset.

We experiment the [Mix-Up augmentation method](https://arxiv.org/abs/1710.09412), and compare results for each model.

The main training script is `train_cifar10.py`. The script takes various parameters, thus we offer suggested parameters, and corresponding results.

### ResNet 

Training a model on ResNet110_v2 can be done with

```
python train_cifar10.py --num-epochs 240 --mode hybrid --num-gpus 2 -j 32 --batch-size 64\
    --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 80,160 --model cifar_resnet110_v2
```

With mixup, the command is

```
python train_cifar10_mixup.py --num-epochs 350 --mode hybrid --num-gpus 2 -j 32 --batch-size 64\
    --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 150,250 --model cifar_resnet110_v2
```

To get results from a different ResNet, modify `--model`.

Results:

| Model        | Accuracy | Mix-Up |
|--------------|----------|--------|
| ResNet20_v1  | 0.9115   | 0.9161 |
| ResNet20_v2  | 0.9117   | 0.9119 |
| ResNet56_v2  | 0.9307   | 0.9414 |
| ResNet110_v2 | 0.9414   | 0.9447 |

Pretrained Model:

| Model        | Accuracy |
|--------------|----------|
| ResNet20_v1  | 0.9160   |
| ResNet56_v1  | 0.9387   |
| ResNet110_v1 | 0.9471   |
| ResNet20_v2  | 0.9130   |
| ResNet56_v2  | 0.9413   |
| ResNet110_v2 | 0.9464   |

### Wide ResNet

Training a model on WRN-28-10 can be done with

```
python train_cifar10.py --num-epochs 200 --mode hybrid --num-gpus 2 -j 32 --batch-size 64\
    --wd 0.0005 --lr 0.1 --lr-decay 0.2 --lr-decay-epoch 60,120,160\
    --model cifar_wideresnet28 --width-factor 10
```

With mixup, the command is

```
python train_cifar10_mixup.py --num-epochs 350 --mode hybrid --num-gpus 2 -j 32 --batch-size 64\
    --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 80,160,240\
    --model cifar_wideresnet28 --width-factor 10
```

To get results from a different WRN, modify `--model` and `--width-factor`.

Results:

| Model        | Accuracy | Mix-Up |
|--------------|----------|--------|
| WRN-16-10    | 0.9527   | 0.9602 |
| WRN-28-10    | 0.9584   | 0.9667 |
| WRN-40-8     | 0.9559   | 0.9620 |

Pretrained Model:

| Model            | Accuracy |
|------------------|----------|
| WideResNet20_v1  | 0.9614   |
| WideResNet56_v1  | 0.9667   |
| WideResNet110_v1 | 0.9673   |

## ImageNet

Here we present examples of training resnet/mobilenet on ImageNet dataset.

The main training script is `train_imagenet.py`. The script takes various parameters, thus we offer suggested parameters, and corresponding results.

### ResNet50_v2

Training a ResNet50_v2 can be done with:

```
python train_imagenet.py --batch-size 64 --num-gpus 4 -j 32 --mode hybrid\
    --num-epochs 120 --lr 0.1 -momentum 0.9 --wd 0.0001\
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
    --num-epochs 200 --lr 0.045 --wd 0.00004\
    --lr-decay 0.98 --lr-decay-period 1 --model mobilenetv2_1.0
```

Results:

| Model            | Top-1 Error | Top-5 Error |
|------------------|-------------|-------------|
| MobileNetV2_1.0  |             |             |

