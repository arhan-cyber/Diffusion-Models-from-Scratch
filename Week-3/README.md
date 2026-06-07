# Week 3 — The Forward Diffusion Process

> **Theme:** Learn how to systematically destroy an image with noise — and why that's useful.
> **Time commitment:** 8–12 hours (more reading than usual)
> **Deliverable due:** End of Week 3 (Friday EOD)

---

## What You'll Build

A **`NoiseScheduler` class** that implements the forward diffusion process. Given a clean image and a timestep `t`, your scheduler produces the correctly-noised image in a single closed-form step.

## Why This Week Matters

This is the most math-heavy week of the program. Don't rush it. If the forward process clicks here, Week 4 (training) feels easy. If it doesn't click here, Week 4 will feel impossible.

## Deliverable Checklist

- [ ] `NoiseScheduler` class with **linear** and **cosine** schedule support
- [ ] Closed-form `q(x_t | x_0)` sampling using the reparameterization trick
- [ ] Visualization script: one image noised at `t = 0, 100, 250, 500, 750, 999`
- [ ] Unit tests verifying `x_T ≈ N(0, I)` (pure Gaussian noise)
- [ ] Plot comparing linear vs cosine SNR curves
- [ ] Code pushed to GitHub with updated README

## Folder Structure

```
week3/
├── README.md
├── scheduler.py       (NoiseScheduler class)
├── visualize.py       (noising trajectory plots)
├── test_scheduler.py  (unit tests)
└── results/
    ├── noising_trajectory.png
    ├── linear_vs_cosine.png
    └── final_noise_distribution.png
```

## Self-Check Questions

1. Why can we sample `x_t` directly from `x_0` in one step instead of iterating `t` times?
2. What's the difference between `β`, `α`, and `ᾱ` (alpha bar)?
3. Why do we need a noise schedule at all — why not just add full noise?
4. What does the reparameterization trick give us?
5. Why does the cosine schedule outperform linear empirically?

## Common Pitfalls

- **Confusing `β`, `α`, and `ᾱ`** → write the definitions on a sticky note before coding
- Numerical instability with `√ᾱ` near `t = T` (ᾱ becomes tiny)
- Forgetting to normalize image data to `[-1, 1]` before adding noise

**Quick links:**
- [Lilian Weng: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) (Sections 1–2 only)
- [DDPM Paper (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) (Sections 1–3.1)
- [Outlier: Diffusion Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg) (watch first if math is intimidating)

---

**Next week:** Train the reverse process. You'll generate your first images from noise. **Mid-program demo day at end of Week 4.**
