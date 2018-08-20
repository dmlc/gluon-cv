## Reproducing LSUN experiments

**With DCGAN:**

```bash
python train_wgan.py --dataset lsun --dataroot [lsun-train-folder] --cuda
```

**With MLP:**

```bash
python main.py --mlp_G --ngf 512
```


## References
["Wasserstein GAN"](https://arxiv.org/abs/1701.07875)