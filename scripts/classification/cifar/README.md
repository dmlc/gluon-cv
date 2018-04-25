# CIFAR10 

Here we present examples of training resnet/wide-resnet on CIFAR10 dataset.

The main training script is `train.py`. The script takes various parameters, thus we offer suggested parameters, and corresponding results.

We also experiment the [Mix-Up augmentation method](https://arxiv.org/abs/1710.09412), and compare results for each model.

## Models

We offer models in `ResNetV1`, `ResNetV2` and `WideResNet`, with various parameters. Following is a list of available pretrained models for certain parameters, and their accuracy on CIFAR10:

| Model            | Accuracy |
|------------------|----------|
| ResNet20_v1      | 0.9160   |
| ResNet56_v1      | 0.9387   |
| ResNet110_v1     | 0.9471   |
| ResNet20_v2      | 0.9158   |
| ResNet56_v2      | 0.9413   |
| ResNet110_v2     | 0.9484   |
| WideResNet16_10  | 0.9614   |
| WideResNet28_10  | 0.9667   |
| WideResNet40_8   | 0.9673   |

## Demo

Before training your own model, you may want to take a look at how it will look like.

Here we provide you a script `demo.py` to load a pre-trained model and predict on an image.

**Execution**

```
python demo --model cifar_resnet110_v2 --input-pic ~/Pictures/demo.jpg
```

**Parameters Explained**

- `--model`: The model to use.
- `--saved-params`: the path to a locally saved model.
- `--input-pic`: the path to the input picture file.

## Training

Training can be done by either `train.py` or `train_mixup.py`.

Training a model on ResNet110_v2 can be done with

```
python train.py --num-epochs 240 --mode hybrid --num-gpus 2 -j 32 --batch-size 64\
    --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 80,160 --model cifar_resnet110_v2
```

With mixup, the command is

```
python train_mixup.py --num-epochs 350 --mode hybrid --num-gpus 2 -j 32 --batch-size 64\
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

by script:

```
python train_mixup.py --num-epochs 450 --mode hybrid --num-gpus 2 -j 32 --batch-size 64 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 150,250 --model cifar_resnet20_v1
```

## Wide ResNet

Training a model on WRN-28-10 can be done with

```
python train.py --num-epochs 200 --mode hybrid --num-gpus 2 -j 32 --batch-size 64\
    --wd 0.0005 --lr 0.1 --lr-decay 0.2 --lr-decay-epoch 60,120,160\
    --model cifar_wideresnet28 --width-factor 10
```

With mixup, the command is

```
python train_mixup.py --num-epochs 350 --mode hybrid --num-gpus 2 -j 32 --batch-size 64\
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

by scripts:

```
python train_mixup.py --num-epochs 500 --mode hybrid --num-gpus 2 -j 32 --batch-size 64 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,200,300 --model cifar_wideresnet16_10
```

**Parameters Explained**

- `--batch-size`: per-device batch size for the training.
- `--num-gpus`: the number of GPUs to use for computation, default is `0` and it means only using CPU.
- `--model`: The model to train. For `CIFAR10` we offer [`ResNet`](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/cifarresnet.py) and [`WideResNet`](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/cifarwideresnet.py) as options.
- `--num-data-workers`/`-j`: the number of data processing workers.
- `--num-epochs`: the number of training epochs.
- `--lr`: the initial learning rate in training. 
- `--momentum`: the momentum parameter.
- `--wd`: the weight decay parameter.
- `--lr-decay`: the learning rate decay factor.
- `--lr-decay-period`: the learning rate decay period, i.e. for every `--lr-decay-period` epochs, the learning rate will decay by a factor of `--lr-decay`.
- `--lr-decay-epoch`: epochs at which the learning rate decay by a factor of `--lr-decay`.
- `--width-factor`: parameters for `WideResNet` model.
- `--drop-rate`: parameters for `WideResNet` model.
- `--mode`: whether to use `hybrid` mode to speed up the training process.
- `--save-period`: for every `--save-period`, the model will be saved to disk.
- `--save-dir`: the directory to save the models.
- `--logging-dir`: the directory to save the training logs.

