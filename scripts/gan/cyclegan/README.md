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

**Visualize The Data**

![sampleA](9_A.jpg "The Input A")

![sampleB](9_B.jpg "The Input B")


## References
["CycleGAN"](https://arxiv.org/abs/1703.10593)
