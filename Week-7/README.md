# Week 7 — Custom Dataset and Final Training Run

> **Theme:** Now make it yours.
> **Time commitment:** 8–12 hours (lots of waiting for training)
> **Deliverable due:** End of Week 7 (Friday EOD)

---

## What You'll Build

Train a **diffusion model on a custom dataset of your choice**. Apply everything from Weeks 1–6 to a domain you actually care about. Your trained model gets pushed to Hugging Face Hub.

## Why This Week Matters

MNIST has been your sandbox. Now you graduate to real data: messy preprocessing, real compute constraints, real choices about augmentation and hyperparameters. This is also where your portfolio piece starts taking shape.

## Pick Your Dataset (Decide by Tuesday)

Choose ONE — switching mid-week kills your training time:

| Dataset | Resolution | Size | Best For |
|---|---|---|---|
| **Pokemon sprites** | 64×64 | ~800 imgs | Quick, fun, overfittable |
| **Smithsonian Butterflies** | 64×64 | ~1000 imgs | Beautiful demos, easy load |
| **Anime faces** | 64×64 | 20k+ | Strong faces, lots of data |
| **CelebA (downsampled)** | 64×64 | 200k | Realistic faces, slow |
| **Quick, Draw! (one category)** | 28×28 | 50k+ | Fast training, sketches |
| **LLD Logo Dataset** | 32×32 | 600k | Design-focused |
| **Your own** | Any | 500+ min | Get mentor approval first |

## Deliverable Checklist

- [ ] Dataset chosen, downloaded, and preprocessed (resize, normalize)
- [ ] Clean data pipeline with augmentation (at minimum, random horizontal flip)
- [ ] **W&B (Weights & Biases) logging** for losses, samples, hyperparameters
- [ ] **≥100 epochs of training** (or until convergence)
- [ ] Sample gallery showing generations at multiple training milestones
- [ ] Model checkpoint pushed to **Hugging Face Hub**
- [ ] Code pushed to GitHub with updated README

## Folder Structure

```
week7/
├── README.md
├── dataset.py         (your custom dataset class)
├── prepare_data.py    (download + preprocess script)
├── train.py           (with W&B logging)
├── sample.py          (final generation script)
├── config.yaml        (hyperparameters)
└── results/
    ├── samples_epoch_25.png
    ├── samples_epoch_50.png
    ├── samples_epoch_100.png
    └── training_curves.png
```

## Self-Check Questions

1. What was the hardest decision you made about your dataset?
2. What augmentations did you use and why? Which would you NOT use (and why)?
3. What batch size + learning rate worked? How did you choose?
4. What would you change if you had 10× the compute?

## Common Pitfalls

- **Underestimating dataset prep time** → often takes longer than coding the model
- **Training too few epochs** → declaring the model "doesn't work" at epoch 20
- **Mode collapse from poor data cleanup** → quality > quantity
- **Not saving intermediate checkpoints** → Colab disconnect = lost progress
- **Switching datasets mid-week** → not enough training time left

## Compute Strategy

- **Colab free tier (T4):** Fine for 64×64 datasets up to ~200 epochs
- **Kaggle Notebooks (P100):** 30 hrs/week free — use as backup
- **Save checkpoints every 10 epochs** — Colab disconnects are not a question of *if* but *when*
- **Start training Tuesday evening** so it can run overnight while you sleep

---

**Next week:** Ship it. Build the demo, write the blog post, present at final showcase.
