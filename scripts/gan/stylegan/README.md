## Reproducing StyleGAN experiments

**Test StyleGAN**
*Test the converted pretrained weights on FFHQ*
```bash
python demo_stylegan.py --path ./stylegan-ffhq-1024px-new-v2.params --gpu_id -1
```

**Generated images from the converted pretrained FFHQ**

![images](sample.jpg "Generated images from the converted pretrained FFHQ")

## References
["A Style-Based Generator Architecture for Generative Adversarial Networks
"](https://arxiv.org/abs/1812.04948)