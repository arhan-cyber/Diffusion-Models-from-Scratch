# Week 7 Implementation Report: Custom Dataset (EMNIST Letters)

This document details the implementation of a class-conditional diffusion model trained on the **EMNIST Letters** dataset using Classifier-Free Guidance (CFG).

---

## 🏗️ Core Architecture & Implementation Details

We graduated from MNIST to EMNIST Letters (~124,800 images, 28x28 grayscale, 26 classes representing English letters A–Z). 

### 1. Data Pipeline & Transposition Correction
EMNIST images are stored natively as transposed (rotated 90 degrees and mirrored). In [dataset.py](./dataset.py), we corrected this by transposing the images back to normal using:
```python
transforms.Lambda(lambda img: img.transpose(PIL.Image.TRANSPOSE))
```
We also mapped original labels (1–26) to standard 0-indexed classes (0–25) so they integrate perfectly with our embedding projection.

### 2. Direction-Aware Data Augmentation
To prevent the model from overfitting without corrupting character semantics, we implemented targeted geometric transforms:
* **Used:** Random rotation of up to $\pm 10$ degrees and random translations of up to $\pm 10\%$.
* **Avoided:** Horizontal/vertical flips, as flipping characters changes their meaning (e.g., 'd' becomes 'b') or creates invalid shapes.

### 3. Centralized YAML Configuration
All hyperparameters are managed in [config.yaml](./config.yaml), decoupling parameters from training scripts.

### 4. Robust Logging Fallbacks
In [train.py](./train.py), W&B logging is checked at runtime. If W&B isn't authenticated, the script logs metrics locally to [results/experiment_log.csv](./results/experiment_log.csv) and saves generation grids locally to the [results/](./results/) directory.

---

## 📊 Results & Visualizations

All generated outputs are stored in the [results/](./results/) directory.

### 1. Training Curves
The model loss was recorded and plotted across epochs:
* View the loss progression: [training_curves.png](./results/training_curves.png)
* Local log file: [experiment_log.csv](./results/experiment_log.csv)

### 2. Generated Milestone Samples
Visual grids showing the generated A–Z letter progression are saved at epoch intervals (every 5 epochs):
* View final generations: [samples_epoch_25.png](./results/samples_epoch_25.png)

---

## 🧠 Self-Check Questions

### 1. What was the hardest decision you made about your dataset?
Deciding how to align and orient EMNIST letters. PyTorch EMNIST returns images flipped and rotated by default. Correcting this rotation/flip in the transform pipeline was crucial to ensure generated letters were readable by humans and displayed properly in our final demo.

### 2. What augmentations did you use and why? Which would you NOT use (and why)?
We used random rotation ($\pm 10^\circ$) and translation ($10\%$) to help the model generalize across different handwriting styles. We strictly avoided horizontal/vertical flips because flipping letters changes their semantic meaning (e.g., flipping a 'b' makes a 'd', or flipping 'E' turns it into an invalid symbol).

### 3. What batch size + learning rate worked? How did you choose?
A batch size of `128` and a learning rate of `1e-4` (Adam optimizer) worked well. These parameters were chosen because they yielded stable convergence in Week 6 and provided a good compute trade-off given the larger size of the EMNIST Letters dataset (~124,800 images).

### 4. What would you change if you had 10× the compute?
* Increase U-Net capacity (e.g. double base channels to 128, add self-attention layers in the bottleneck and decoder).
* Train for 100+ epochs to fully converge.
* Upscale the dataset to $64\times64$ resolution.
* Use multi-GPU training with PyTorch DDP.
