# Week 2 — Convolutional Networks and the UNet

> **Theme:** Build the architecture that powers every diffusion model.
> **Time commitment:** 8–12 hours
> **Deliverable due:** End of Week 2 (Friday EOD)

---

## What You'll Build

A **UNet from scratch** in PyTorch, trained as a denoising autoencoder on MNIST or CIFAR-10. Take clean images, add Gaussian noise, train the UNet to reconstruct them.

## Why This Week Matters

The UNet is the architectural backbone of every diffusion model — from Stable Diffusion to DALL-E to your final project. Build it now, understand every skip connection, every dimension. You'll thank yourself in Week 4.

## Deliverable Checklist

- [ ] Modular building blocks: `DoubleConv`, `Down`, `Up`
- [ ] Configurable depth and channel count (passed via constructor)
- [ ] Skip connections correctly implemented (concatenation, not addition)
- [ ] Trained on MNIST or CIFAR-10 denoising task
- [ ] Visualization: noisy input → denoised output (every N epochs)
- [ ] Code pushed to GitHub with updated README

## Folder Structure

```
week2/
├── README.md
├── model.py           (UNet implementation)
├── blocks.py          (DoubleConv, Down, Up)
├── train.py           (denoising training loop)
├── dataset.py         (adds noise on-the-fly)
└── results/
    ├── samples_epoch_10.png
    ├── samples_epoch_50.png
    └── loss_curve.png
```

## Self-Check Questions

1. Why are skip connections in a UNet important? What happens without them?
2. What's the role of the bottleneck in a UNet?
3. Difference between transposed convolution and bilinear upsampling?
4. Given input 64×64, kernel 3×3, stride 2, padding 1 — what's the output size?

## Common Pitfalls

- **Off-by-one dimension errors** on skip-connection concatenation → use `padding=1` with `kernel_size=3` and same-size convs throughout
- Output channels not matching input channels (denoising should be 1→1 for MNIST, 3→3 for CIFAR)
- Bottleneck too narrow for the dataset complexity

**Quick links:**
- [CS231n: CNN Module](https://cs231n.github.io/convolutional-networks/)
- [UNet Paper (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)
- [Aman Arora: UNet in 60 Lines](https://amaarora.github.io/posts/2020-09-13-unet.html)

---

**Next week:** The forward diffusion process — how to systematically destroy an image with noise, and why that's actually useful.
