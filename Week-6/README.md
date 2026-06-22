# Week 6 — Conditional Generation and Classifier-Free Guidance

> **Theme:** Don't just generate random digits. Generate the digit you want.
> **Time commitment:** 8–12 hours
> **Deliverable due:** End of Week 6 (Friday EOD)

---

## What You'll Build

Extend your model to support **class-conditional generation** with **classifier-free guidance (CFG)** — the same technique behind Stable Diffusion, DALL-E, and every modern text-to-image model.

## Why This Week Matters

Random generation is a parlor trick. Controlled generation is what makes diffusion models useful. CFG is the magic that lets you crank up "how strongly should this look like a 7" via a single slider. By the end of this week, you'll understand how every modern generative AI tool actually works.

## Deliverable Checklist

- [ ] **Label embeddings** added alongside timestep embeddings
- [ ] **Random label dropout** (10–20%) during training for CFG support
- [ ] CFG sampling with adjustable guidance scale `w`
- [ ] Sample grid: **same noise vector → different classes** (proves conditioning works)
- [ ] Sample grid: **same class with `w = 1, 3, 5, 10`** (proves CFG works)
- [ ] Verify ~15% of training batches use the null label (print a debug line)
- [ ] Code pushed to GitHub with updated README

## Folder Structure

```
week6/
├── README.md
├── model.py           (UNet with label embeddings)
├── train.py           (with label dropout)
├── sample.py          (CFG sampling)
└── results/
    ├── conditional_grid.png      (same noise, different classes)
    ├── guidance_scale_sweep.png  (same class, varying w)
    └── unconditional_vs_cfg.png
```

## Self-Check Questions

1. Why do we randomly drop conditioning during training? What happens if we don't?
2. Write the CFG sampling formula. Why does the sign matter?
3. Why does increasing guidance scale improve quality but reduce diversity?
4. What's the difference between classifier guidance (old) and classifier-free guidance (new)?

## Common Pitfalls

- **Forgetting to randomly drop conditioning during training** → CFG produces nonsense at inference (the #1 bug this week)
- **Wrong sign on the CFG formula** → samples look worse with higher guidance instead of better
- **Overly high guidance scales** (w > 15) → oversaturated, artifact-heavy images
- Not verifying conditioning actually works before adding CFG → debug in two stages

## Debugging Strategy

Get conditioning working FIRST without CFG (just label embeddings, no dropout). Verify "same noise → different classes" gives recognizably different outputs. THEN add label dropout. THEN add CFG. Three separate debugging stages.

---

**Next week:** Your own dataset. Make it personal.
