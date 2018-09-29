## Reproducing Cycle GAN experiments


**Download horse2zebra dataset**
```bash
bash ./download_dataset.sh horse2zebra
```

**Monitoring loss values and images during training**
```bash
pip install mxboard
tensorboard --logdir=./logs --host=127.0.0.1 --port=8888
```
Details about mxboard is in [mxboard](https://github.com/awslabs/mxboard)

**Train Cycle GAN**
```bash
python train_cgan.py --dataroot ./horse2zebra
```
![images](images.png "images during training")
The meaning of those images are :
| real_A(from dataset) | fake_B(generate from real_A) | rec_A(reconstruct from fake_B) | idt_A(generate from real_B) |
| real_B(from dataset) | fake_A(generate from real_A) | rec_B(reconstruct from fake_A) | idt_B(generate from real_A) |