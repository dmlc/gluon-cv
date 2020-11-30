# COOT
by Simon Ging\*, Mohammadreza Zolfaghari\*, Hamed Pirsiavash\*, and Thomas Brox.

(\*) equal technical contribution.

## Introduction
In this we release code and models from the paper [COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning](https://arxiv.org/pdf/2011.00597v1.pdf).
Overview of the model: 


## Installation

Make sure you have `Python>=3.6` installed on your machine. For installing dependecies follow the instruction of GluonCV.


## Training
You can train the model on your dataset of interest. 

```
python3 -m  --dataset youcook2
```

## Evaluation
You can evaluate the model with following command. 

```
python3 -m  --dataset youcook2
```

## Expected results

In this table we closely follow experiments from the coot paper and report results
that were achieved by running this code on AWS machine with one Tesla T4 GPU.

## Citation

```
@inproceedings{ging2020coot,
  title={COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning},
  author={Simon Ging and Mohammadreza Zolfaghari and Hamed Pirsiavash and Thomas Brox},
  booktitle={Conference on Neural Information Processing Systems},
  year={2020}
}
```