# [A Comprehensive Study of Deep Video Action Recognition](https://arxiv.org/abs/2012.06567)

We provide a series of tutorials for new comers to this field, including this survey paper, the [CVPR2020 video tutorial](https://bryanyzhu.github.io/videomodeling.github.io/), the [YouTube videos](https://www.youtube.com/watch?v=Jwt0Wtlv_uo&list=PLGCZZzK2R0X6RQiQrbShUULsbF1qeC17d) and the implementations in GluonCV (both PyTorch and MXNet).


## Dataset

We do not own or manage the data, but to ensure reproducibility, please use the copies available in here: [Kinetics400](https://academictorrents.com/details/184d11318372f70018cf9a72ef867e2fb9ce1d26) and [Kinetcs700](https://academictorrents.com/details/49f203189fb69ae96fb40a6d0e129949e1dfec98).


## Training/Evaluation

We use our PyTorch implementations to report numbers in the survey paper. Configurations for both training and evaluation of the models can be found in the yaml files under folder ``configuration``. Note that, at this moment, the model weights of ``r2plus1d_v2_resnet152_kinetics400``, ``ircsn_v2_resnet152_f32s2_kinetics400`` and ``TPN family`` are ported from original [VMZ](https://github.com/facebookresearch/VMZ) and [TPN](https://github.com/decisionforce/TPN) repository. You may ignore the training config of these models.


## Citation

If you feel our code or models helps in your research, kindly cite our papers:

```
@article{zhu_arxiv2020_comprehensiveVideo,
  title={A Comprehensive Study of Deep Video Action Recognition},
  author={Yi Zhu, Xinyu Li, Chunhui Liu, Mohammadreza Zolfaghari, Yuanjun Xiong, Chongruo Wu, Zhi Zhang, Joseph Tighe, R. Manmatha, Mu Li},
  journal={arXiv preprint arXiv:2012.06567},
  year={2020}
}
```
