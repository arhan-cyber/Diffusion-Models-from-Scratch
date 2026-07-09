# Week 7 Implementation Plan: EMNIST Letters Diffusion Model

This document outlines the detailed implementation plan for training a class-conditional diffusion model on the **EMNIST letters** dataset using classifier-free guidance (CFG).

## 1. Project Specifications

* **Dataset:** EMNIST `letters` split (~124,800 images, 28x28 grayscale, 26 classes representing English letters A–Z).
* **Model Type:** Class-conditional U-Net with Classifier-Free Guidance (CFG).
* **Training Platform:** Local GPU (NVIDIA CUDA, Apple Silicon MPS, or AMD ROCm) via command-line execution.
* **Duration:** 25 epochs (configurable).
* **Logging System:** Optional Weights & Biases (W&B) integration with a local fallback (CSV log and local image generation).

---

## 2. Directory Structure

We will create the following files inside the [Week-7/](file:///C:/Users/Arhan/Projects/soc/Diffusion-Models-from-Scratch/Week-7) directory:

```
Week-7/
├── README.md            (Existing instructions)
├── Week_7_7.pdf         (Existing curriculum)
├── config.yaml          (Hyperparameter and training configuration)
├── prepare_data.py      (Data download and verification script)
├── dataset.py           (Custom Dataset wrapper mapping labels 1-26 to 0-25 with safe augmentations)
├── blocks.py            (UNet layers, adapted from Week 6)
├── embeddings.py        (Sinusoidal timestep embeddings, adapted from Week 6)
├── model.py             (Conditional UNet model, adapted from Week 6)
├── diffusion.py         (Forward and reverse DDPM processes with CFG sampling, adapted from Week 6)
├── train.py             (Training loop with optional W&B and CSV fallback)
├── sample.py            (Inference script for custom letter generation)
└── results/             (Directory for saved samples and training curves)
```

---

## 3. Detailed Component Designs

### A. Data Preprocessing & Augmentation ([dataset.py](file:///C:/Users/Arhan/Projects/soc/Diffusion-Models-from-Scratch/Week-7/dataset.py))
* **Preprocessing:** Normalize images to $[-1, 1]$ range.
* **Label Mapping:** Convert EMNIST's 1-indexed labels ($1$ to $26$) to 0-indexed class IDs ($0$ to $25$) by subtracting $1$.
* **Augmentations:** 
  * **Safe Rotations:** Random rotation of up to $\pm 10$ degrees.
  * **Safe Translations:** Random affine translation of up to $\pm 10\%$.
  * **No Flips:** Avoid horizontal or vertical flips to prevent converting letters into other letters (like 'b' to 'd') or invalid symbols.

### B. Configuration ([config.yaml](file:///C:/Users/Arhan/Projects/soc/Diffusion-Models-from-Scratch/Week-7/config.yaml))
Defines hyperparameters in a central location:
```yaml
training:
  epochs: 25
  batch_size: 128
  learning_rate: 0.0001
  noise_steps: 1000
  sample_every: 5
  checkpoint_every: 5
  use_wandb: false  # Can be overridden via CLI --wandb
  wandb_project: "diffusion-letters"
model:
  base_channels: 64
  depth: 3
  emb_dim: 256
```

### C. Robust Training ([train.py](file:///C:/Users/Arhan/Projects/soc/Diffusion-Models-from-Scratch/Week-7/train.py))
* Checks if `wandb` is installed and if the user is authenticated. If enabled but credentials aren't set, it gracefully notifies you and defaults to local logging.
* Logs loss, learning rate, and epoch durations to `results/experiment_log.csv`.
* Saves checkpoints under `checkpoints/` periodically.
* Generates sample grids of letters A–Z at regular epoch intervals.

### D. Letter Inference ([sample.py](file:///C:/Users/Arhan/Projects/soc/Diffusion-Models-from-Scratch/Week-7/sample.py))
* Loads the trained weights and generates specific letters from A–Z.
* Allows configuring the Classifier-Free Guidance scale (`w` or `cfg_scale`) to control generation fidelity.
