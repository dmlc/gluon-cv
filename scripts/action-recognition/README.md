# Action Recognition[1]
[GluonCV Model Zoo](https://gluon-cv.mxnet.io/model_zoo/action_recognition.html)

If you feel our code or models helps in your research, kindly cite our papers:

```
@article{zhu_arxiv2020_comprehensiveVideo,
  title={A Comprehensive Study of Deep Video Action Recognition},
  author={Yi Zhu, Xinyu Li, Chunhui Liu, Mohammadreza Zolfaghari, Yuanjun Xiong, Chongruo Wu, Zhi Zhang, Joseph Tighe, R. Manmatha, Mu Li},
  journal={arXiv preprint arXiv:2012.06567},
  year={2020}
}
```

## PyTorch Tutorial

### [How to train?](https://cv.gluon.ai/build/examples_torch_action_recognition/finetune_custom.html)

```
python train_ddp_pytorch.py --config-file ./configuration/XXX.yaml
```

If multi-grid training is needed,
```
python train_ddp_shortonly_pytorch.py --config-file ./configuration/XXX.yaml
```
Note that we only use short-cycle here because it is stable and applies to a range of models.


### [How to evaluate?](https://cv.gluon.ai/build/examples_torch_action_recognition/demo_i3d_kinetics400.html)

```
# Change PRETRAINED to True if using our pretraind model zoo
python test_ddp_pytorch.py --config-file ./configuration/XXX.yaml
```

### [How to extract features?](https://cv.gluon.ai/build/examples_torch_action_recognition/extract_feat.html)

```
python feat_extract_pytorch.py --config-file ./configuration/XXX.yaml
```

### [How to get speed measurement?](https://cv.gluon.ai/build/examples_torch_action_recognition/speed.html)

```
python get_flops.py --config-file ./configuration/XXX.yaml
python get_fps.py --config-file ./configuration/XXX.yaml
```


## MXNet Tutorial


### [How to train?](https://cv.gluon.ai/build/examples_action_recognition/dive_deep_i3d_kinetics400.html)
MXNet codebase adopts argparser, hence requiring many arguments. Please check [model zoo page](https://cv.gluon.ai/model_zoo/action_recognition.html) for detailed training command.

```
python train_recognizer.py
```

### [How to evaluate?](https://cv.gluon.ai/build/examples_action_recognition/demo_i3d_kinetics400.html)

```
python test_recognizer.py
```

### [How to extract features?](https://cv.gluon.ai/build/examples_action_recognition/feat_custom.html)

```
python feat_extract.py
```

### [How to do inference on your own video?](https://cv.gluon.ai/build/examples_action_recognition/demo_custom.html)

```
python inference.py
```

## MXNet calibration
Please check out [CALIBRATION.md](https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/action-recognition/CALIBRATION.md) for more information on INT8 model calibration and inference.

## Reproducing our arXiv survey paper

Please check out [ARXIV.md](https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/action-recognition/ARXIV.md) for more information on how to get the same dataset and how to reproduce all the methods in our model zoo.
