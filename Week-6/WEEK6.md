# Week 6 Implementation Report: Conditional Generation & CFG

This document details the implementation of **class-conditional generation** using **Classifier-Free Guidance (CFG)**, trained on the MNIST dataset.

---

## 🏗️ Core Architecture & Implementation Details

We extended the unconditional DDPM/DDIM architecture into a conditional generator using three main steps:

### 1. Class Conditioning in UNet
In [model.py](file:///C:/Users/Arhan/Projects/soc/Diffusion-Models-from-Scratch/Week-6/model.py), we added a learnable embedding layer:
```python
self.class_embedding = nn.Embedding(num_classes + 1, emb_dim)
```
* **Embedding Dimension:** Same as the timestep embedding dimension (`emb_dim=256`).
* **Null Class:** The layer size is `num_classes + 1` (index `10`), allowing class `10` to act as the unconditional/null token.
* **Embedding Fusion:** The timestep embedding and class embedding are added together before being injected into the ResNet/DoubleConv layers.

### 2. Random Label Dropout
In [train.py](file:///C:/Users/Arhan/Projects/soc/Diffusion-Models-from-Scratch/Week-6/train.py), we implemented random label dropout to enable joint training of conditional and unconditional diffusion models:
```python
dropout_mask = torch.rand(labels.shape[0], device=device) < 0.15
cond_labels = torch.where(dropout_mask, torch.tensor(10, device=device), labels)
```
* **Dropout Rate:** 15% of the labels are replaced with the null class (`10`).
* **Purpose:** This forces the model to learn both conditional prediction $\epsilon_{\theta}(x_t, t, c)$ and unconditional prediction $\epsilon_{\theta}(x_t, t, \emptyset)$ simultaneously.

### 3. Classifier-Free Guidance (CFG) Sampling
In [diffusion.py](file:///C:/Users/Arhan/Projects/soc/Diffusion-Models-from-Scratch/Week-6/diffusion.py), the reverse sampling process is adjusted to perform a dual forward pass when a guidance scale $w > 0.0$ is provided:
```python
predicted_noise_cond = model(x, t, labels)
null_labels = torch.full_like(labels, model.num_classes)
predicted_noise_uncond = model(x, t, null_labels)

# Combine:
predicted_noise = (1.0 + cfg_scale) * predicted_noise_cond - cfg_scale * predicted_noise_uncond
```

---

## 📊 Results & Visualizations

All results are saved in the [results/](file:///C:/Users/Arhan/Projects/soc/Diffusion-Models-from-Scratch/Week-6/results/) directory.

### 1. Training Loss Curve
The model was trained for 15 epochs on MNIST. The loss history is documented below:
* Check out the [loss_curve.png](file:///C:/Users/Arhan/Projects/soc/Diffusion-Models-from-Scratch/Week-6/results/loss_curve.png)

### 2. Conditional Generation (Same Noise $\rightarrow$ Different Classes)
We generated images using the **exact same initial noise vectors** across digits 0–9. This proves the class conditioning modifies the semantic content of the image while preserving the spatial layout derived from the seed noise.
* View the [conditional_grid.png](file:///C:/Users/Arhan/Projects/soc/Diffusion-Models-from-Scratch/Week-6/results/conditional_grid.png)

### 3. Guidance Scale Sweep ($w = 1, 3, 5, 10$)
We generated digit `7` using a shared set of noise vectors across four guidance scales $w \in \{1.0, 3.0, 5.0, 10.0\}$.
* As $w$ increases, the sharpness and typical features of the digit `7` become significantly more pronounced, proving that the guidance scale successfully controls the strength of conditioning.
* View the [guidance_scale_sweep.png](file:///C:/Users/Arhan/Projects/soc/Diffusion-Models-from-Scratch/Week-6/results/guidance_scale_sweep.png)

### 4. Unconditional vs. CFG Sampling
Comparison between pure unconditional generation ($w=0$, using the null label) and guided conditional generation ($w=3.0$).
* View the [unconditional_vs_cfg.png](file:///C:/Users/Arhan/Projects/soc/Diffusion-Models-from-Scratch/Week-6/results/unconditional_vs_cfg.png)
