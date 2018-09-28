<img src='./img/horse2zebra.gif' align="right" width=384>

<br><br><br>

## A Mini Implementation Of Cyclagan


**Prerequisites**
1. Python 3.6+
2. Gluoncv
3. Mxnet


**Download  Dataset**
```bash
cd ./datasets
python extract_data.py
```

**Run The Model**
```bash
python Cyclegan.py --num_epoch 50 --lr 0.00001
```

**Demo**

<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="800"/>


## References
["CycleGAN"](https://arxiv.org/abs/1703.10593)
