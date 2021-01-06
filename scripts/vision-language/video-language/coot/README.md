# COOT
COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning 
by Simon Ging\*, Mohammadreza Zolfaghari\*, Hamed Pirsiavash\*, and Thomas Brox.

(\*) equal technical contribution.

## Introduction
In this we release code and models from the paper [COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning](https://arxiv.org/pdf/2011.00597v1.pdf).

Many real-world video-text tasks involve different levels of granularity, such as frames and words, clip and sentences or videos and paragraphs, each with distinct semantics. In this paper, we propose a Cooperative hierarchical Transformer (COOT) to leverage this hierarchy information and model the interactions between different levels of granularity and different modalities. The method consists of three major components: an attention-aware feature aggregation layer, which leverages the local temporal context (intra-level, e.g., within a clip), a contextual transformer to learn the interactions between low-level and high-level semantics (inter-level, e.g. clip-video, sentence-paragraph), and a cross-modal cycle-consistency loss to connect video and text. The resulting method compares favorably to the state of the art on several benchmarks while having few parameters.

## Installation

Make sure you have `Python>=3.6` installed on your machine. For installing dependecies follow the instruction of GluonCV.

## Download Pre-computed features
Download the following data with  ``` wget ``` and put all in a folder (e.g. `/data/youcook2`). Then set the ``` DATA_PATH ``` in the config file to this path (e.g `DATA_PATH: /data/youcook2`).

- Youcook2
  - Meta data [Link](https://yzaws-data-log.s3.amazonaws.com/shared/COOT/youcook2/meta_100m.json)
  - Text features [Link](https://yzaws-data-log.s3.amazonaws.com/shared/COOT/youcook2/text_default.h5)
  - Text lengths [Link](https://yzaws-data-log.s3.amazonaws.com/shared/COOT/youcook2/text_lens_default.json)
  - Video features [Link](https://yzaws-data-log.s3.amazonaws.com/shared/COOT/youcook2/video_feat_100m.h5)
## Training
You can train the model on your dataset of interest. 

```
python3 -m scripts.vision-language.video-language.coot.train_pytorch --config-file scripts/vision-language/video-language/coot/configuration/youcook2.yaml

```

## Expected results

In this table we closely follow experiments from the coot paper and report results
that were achieved by running this code on AWS machine with one Tesla T4 GPU.

### Results on Youcook2 dataset with HowTo100m features
| Model | Paragraph->Video R@1 | R@5  | R@10 | MR  | Sentence->Clip R@1 | R@5  | R@10 | MR  |
| ----- | -------------------- | ---- | ---- | --- | ------------------ | ---- | ---- | --- |
| COOT  | 78.3                 | 96.2 | 97.8 | 1   | 16.9               | 40.5 | 52.5 | 9   |

## Citation

```
@inproceedings{ging2020coot,
  title={COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning},
  author={Simon Ging and Mohammadreza Zolfaghari and Hamed Pirsiavash and Thomas Brox},
  booktitle={Conference on Neural Information Processing Systems},
  year={2020}
}
```

