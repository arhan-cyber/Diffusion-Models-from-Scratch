# Week 4 — Training the Reverse Process

> **Theme:** Generate your first images. Likely blurry. Likely magical anyway.
> **Time commitment:** 8–12 hours
> **Deliverable due:** End of Week 4 (Friday EOD) — **There may be a DEMO DAY**

---

## What You'll Build

A **fully working DDPM** trained on MNIST or Fashion-MNIST. By Sunday, you generate recognizable digits/clothes from pure Gaussian noise.

## Why This Week Matters

This is the **single most important week** of the program. If your DDPM works on MNIST by Sunday, the rest of the program is downhill. Everything from Week 5 onwards builds on this foundation.

## Deliverable Checklist

- [ ] Sinusoidal **timestep embeddings** integrated into UNet blocks
- [ ] Training loop following **Algorithm 1** from DDPM paper
- [ ] Sampling loop following **Algorithm 2** from DDPM paper
- [ ] Recognizable samples generated on MNIST/Fashion-MNIST
- [ ] Sample grid saved every N epochs (visual progress)
- [ ] Training loss curve plotted
- [ ] README updated with results gallery
- [ ] **Present at mid-program demo on Sunday**

## Folder Structure

```
week4/
├── README.md
├── model.py           (UNet + timestep embeddings)
├── diffusion.py       (training + sampling logic)
├── train.py           (training script)
├── sample.py          (generation script)
├── embeddings.py      (sinusoidal timestep encoding)
├── checkpoints/
│   └── ddpm_mnist.pt
└── results/
    ├── samples_epoch_10.png
    ├── samples_epoch_50.png
    ├── samples_final.png
    └── loss_curve.png
```

## Self-Check Questions

1. Walk through one iteration of Algorithm 1 (training) line by line.
2. Walk through one iteration of Algorithm 2 (sampling) line by line.
3. During training we predict noise. During sampling, how do we use that to denoise?
4. Why do we predict noise instead of directly predicting `x_{t-1}`?
5. How do timestep embeddings get injected into UNet blocks?

## Common Pitfalls

- **Forgetting timestep embeddings entirely** → model trains but generates noise
- Sampling loop **sign errors** → samples diverge to NaN
- **Too few training epochs** → MNIST needs 50–100 epochs minimum (don't declare it "broken" at epoch 10)
- Forgetting `model.eval()` and `torch.no_grad()` during sampling

**Quick links:**
- [Hugging Face: The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) ⭐ **Essential**
- [HF Diffusion Course: From Scratch](https://huggingface.co/learn/diffusion-course/en/unit1/3)
- [DDPM Paper Sections 3.2 and 4](https://arxiv.org/abs/2006.11239)
- [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) (read, don't copy)

## 

Prepare a **5 slide presentation**:
- Your best generated samples
- Biggest debugging challenge of the past 4 weeks
- One thing you want to improve in the second half

---

**Next week:** DDIM sampling — make your generation 10–100× faster without retraining.
