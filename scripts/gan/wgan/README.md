## Reproducing LSUN experiments

**Download LSUN dataset**
```bash
cd ../../../scripts/datasets/
python2.7 lsun.py -c bedroom
```

**With DCGAN:**

```bash
python train_wgan.py --dataset lsun --dataroot [lsun-train-folder] --cuda
```

**With MLP:**

```bash
python main.py --mlp_G --ngf 512
```

**generate fake samples after 400000 epoch**
![gensample](fake_samples_400000.png "fake samples in 400000 epoch")

## References
["Wasserstein GAN"](https://arxiv.org/abs/1701.07875)