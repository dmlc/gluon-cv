Training Your First Classification Model on CIFAR10
=============

[`CIFAR10`](https://www.cs.toronto.edu/~kriz/cifar.html) is a labeled dataset of tiny (32x32) images, collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. It is widely used as a benchmark in conputer vision research.

In this tutorial, we will demonstrate how to use `Gluon` to train a model from scratch and reproduce the performance from papers. Specifically, we offer a script to prepare the `CIFAR10` dataset and train a `ResNet` model at [scripts/classification/train_cifar10.py](https://github.com/dmlc/gluon-vision/blob/master/scripts/classification/train_cifar10.py).

In the following content, we will  

- explain the parameters in the script
- train the model with a set of parameters
- demonstrate how to restart the training
- plot the training history
- predict on new images with saved model


## Training Your First Model

Now we can train our first model with 

```
python train_cifar10.py --num-epochs 240 --mode hybrid --num-gpus 2 -j 32 --batch-size 64\
    --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 80,160 --model cifar_resnet20_v2
```

Here we train a `ResNet20_V2` model on `CIFAR10` for 240 epochs on two GPUs. The batch size for each GPU is 64, thus the total batch size is 128. We decay the learning rate by a factor of 10 at the 80th and 160th epoch.

The dataset and the model are relatively small, thus it won't take too long to train the model. After the training, the accuracy is expect to be around 91%. To get a better accuracy, we can train a `ResNet110_V2` model instead by setting `--model cifar_resnet20_v2` in the command. With `ResNet110_V2`, we expect the accuracy to be around 94%.

## Resume Training From Model

It is always good to have the ability to resume the previous training process. For example, the server shutdown unexpectedly after training the 100-th epoch, now we want to resume the training from the 100-th epoch and continue for another 140 epochs. In the provided script, we can resume the training process from it by

```
python train_cifar10.py --num-epochs 140 --mode hybrid --num-gpus 2 -j 32 --batch-size 64\
    --wd 0.0001 --lr 0.01 --lr-decay 0.1 --lr-decay-epoch 60 --model cifar_resnet20_v2\
    --resume-from params/resnet_100.params
```

Notice that we add `--resume-from params/resnet_100.params` to set the saved model file. Besides, we also need to modify `--num-epochs`, `--lr` and `--lr-decay-epoch` to make sure the total steps and the learning rate are set correctly.

We need to point it out that this is not equivalent to a full training procedure, because the status (e.g. momentum) in the optimizer cannot be resumed.

## Review the Result

The training a deep learning model is usually a trial-and-error process. A good way to inspect the result is to have a plot:

![]()

This is a plot generated from the following command:

```
```

We see that the issue could be not enough epochs. We then change to

```
```

and observe that

![]()

## Predict with Trained Model

With so much efforts being put into the model training, we can now play around with some pictures not included in the original `CIFAR10` dataset. Although people usually don't use a model trained on `CIFAR10` for this job, we still want to demonstrate how well this model behaves.

We offer pretrained models on `CIFAR10`, and it is accessible by simply setting `pretrained=True`.


The following is a picture of frog from Google Image.

![]()

Then we process the image by resize it to be 32x32 and normalize it.

```
```

And finally we put it through our model, the result looks like:

```
```


It works!
