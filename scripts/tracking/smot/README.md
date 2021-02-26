# [SMOT](https://arxiv.org/abs/2010.16031)

**SMOT: Single-Shot Multi Object Tracking**

by Wei Li, Yuanjun Xiong, Shuo Yang, Siqi Deng and Wei Xia


## Introduction

In this, we release code and models from the paper [Single-Shot Multi Object Tracking (SMOT)](https://arxiv.org/abs/2010.16031), to perform multi-object tracking. SMOT is a new tracking framework that converts any single-shot detector (SSD) model into an online multiple object tracker. **You can track any object as long as the detector can recognize the category, no training is required.** We also want to point out that, SMOT is very efficient, its runtime is close to the runtime of the chosen detector.


## Installation

For installation, follow the instruction of GluonCV. Optionally, if you would like to perform evaluation on MOT17 data and obtain metrics, please install `motmetrics` as `pip install motmetrics`.


## Demo on a video

Want to get multi-person tracking results on a video? Try this

```
python demo.py VIDEOFILE --input-type video --network-name ssd_512_mobilenet1.0_coco --use-pretrained --custom-classes person
```

- `VIDEOFILE` is the path to your demo video. It can also be the path to a folder of a sequence of images, but in this case, please remember to change the following `--input-type` to `images`.

- `--input-type` indicates the input type. It can be either `video` or `images`.

- `--network-name` is the detector you want to use. We use `ssd_512_mobilenet1.0_coco` here due to its good tradeoff between tracking accuracy and efficiency. You can find other detetors in our model zoo that suits your use case.

- `--use-pretrained` indicates you want to use the pretrained weights from GluonCV model zoo. If you don't specify `--use-pretrained`, please use `--param-path` to pass in your detector pretrained weights.

- `--custom-classes` indicates which object category you want to track. You can also track multiple categories at the same time, e.g., `--custom-classes person car dog`.

- The tracking visualization results will be saved to `./smot_vis`, but you can change the directory by specifing `--save-path`.


## Eval on exisiting datasets

We provide evaluation code on MOT17 to have some numerical results in case you need a comparison.

First of all, please download MOT17 data from its [official website](https://motchallenge.net/data/MOT17.zip).

Second, assume you unzip the data into `MOT17_DATA` folder, please proprocess the data into our format for easier evaluation,

```
python preprocess.py --mot-folder MOT17_DATA --mot-save-folder ./jsons/ --mot-save-npy ./npys/
```

The transformed data in json format will be saved to `./jsons` and its npy format will be saved to `./npys`. We only need the npy format data for evaluation.


Third, generate predictions using SMOT,

```
python demo.py MOT17_DATA/MOT17-02-FRCNN/img1/ --input-type images --network-name ssd_512_mobilenet1.0_coco --use-pretrained --custom-classes person --use-motion --save-path ./pred
```

Since MOT data is stored in image sequences, we need to change `input-type` to `images`. `use-motion` can be used optionally to improve the tracking performance. The predictions will be saved in npy format to the path `./pred`.


Finally, run evaluation on the predictions and ground-truth, the results will be printed to console in a table.

```
python eval.py --gt-dir ./npys/ --pred-dir ./pred/
```

For more information, please check `eval.py`.



## Citation

```
@inproceedings{li2020smot,
  title={SMOT: Single-Shot Multi Object Tracking},
  author={Wei Li, Yuanjun Xiong, Shuo Yang, Siqi Deng, Wei Xia},
  booktitle={arXiv preprint arXiv:2010.16031},
  year={2020}
}
```
