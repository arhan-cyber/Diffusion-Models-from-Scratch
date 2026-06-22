# Week 5 — Faster Sampling with DDIM

> **Theme:** 1000 sampling steps is too many. Let's cut it to 50 with no quality loss.
> **Time commitment:** 8–12 hours
> **Deliverable due:** End of Week 5 (Friday EOD)

---

## What You'll Build

A **`DDIMScheduler`** added to your Week 4 model. Sample 10–100× faster from your existing trained DDPM — no retraining needed. Then benchmark the speed/quality trade-off rigorously.

## Why This Week Matters

Sampling 1000 timesteps per image isn't practical for real-world demos (your Hugging Face Space would time out). DDIM is the same idea that powers production diffusion models. It also teaches you that *sampling* and *training* are independent design choices.

## Folder Structure

```
week5/
├── README.md
├── ddim_scheduler.py  (new DDIM implementation)
├── benchmark.py       (timing + comparison script)
├── benchmark.md       (your findings)
└── results/
    ├── ddpm_1000_steps.png
    ├── ddim_10_steps.png
    ├── ddim_25_steps.png
    ├── ddim_50_steps.png
    ├── ddim_100_steps.png
    └── timing_comparison.png
```

## Self-Check Questions

1. How does DDIM use the same trained model as DDPM but sample faster?
2. What does the `eta` parameter control? What does `eta=0` mean? `eta=1`?
3. How does DDIM choose its smaller timestep subset from the original schedule?
4. What's the trade-off: when would you prefer DDPM over DDIM?

## Common Pitfalls

- **Mixing up the DDIM update equation** with DDPM's stochastic one → use Equation 12 from the paper
- **Wrong timestep subset** → DDIM picks evenly-spaced timesteps from the original schedule
- Forgetting that `eta=1` reduces to DDPM → use this as a debugging sanity check
- Comparing samples from different random seeds → fix the seed across all benchmark configs

## Bonus Challenge

Try **latent interpolation**: pick two random noise vectors and visualize the smooth path between their generations. Only possible because DDIM is deterministic — beautiful demo material for Week 8.

---

**Next week:** Conditional generation and classifier-free guidance — control what your model generates.
