# Week 5 — Faster Sampling with DDIM

This folder contains a complete implementation of a **`DDIMScheduler`** that performs deterministic and stochastic sampling from a pre-trained DDPM model.

## Implementation Details

### `DDIMScheduler` ([ddim_scheduler.py](file:///C:/Users/Arhan/Projects/soc/Diffusion-Models-from-Scratch/Week-5/ddim_scheduler.py))
The scheduler supports:
- **Timestep Subsetting**: Evenly-spaced steps chosen from $[0, T-1]$.
- **Configurable $\eta$ (eta)**:
  - $\eta = 0.0$: Fully deterministic DDIM sampling (no added noise).
  - $\eta = 1.0$: Recovers stochastic DDPM sampling behavior.
- **Update Equation**:
  $$x_{t_{i-1}} = \sqrt{\alpha_{t_{i-1}}} \left( \frac{x_{t_i} - \sqrt{1 - \alpha_{t_i}} \epsilon_\theta(x_{t_i}, t_i)}{\sqrt{\alpha_{t_i}}} \right) + \sqrt{1 - \alpha_{t_{i-1}} - \sigma_{t_i}^2} \epsilon_\theta(x_{t_i}, t_i) + \sigma_{t_i} \epsilon$$

---

## Self-Check Answers

### 1. How does DDIM use the same trained model as DDPM but sample faster?
DDIM modifies the inference trajectory, not the training objective. Since the forward process $q(x_t|x_0)$ has the same marginal distribution under both DDPM and DDIM, a UNet trained to predict the noise $\epsilon$ added to $x_0$ to get $x_t$ can be used under any inference scheduler that maintains the same marginal forward step variance. By mapping non-Markovian forward steps, DDIM allows taking larger step sizes (skipping timesteps) without introducing massive errors in the variance calculation.

### 2. What does the $\eta$ parameter control? What does $\eta=0$ mean? $\eta=1$?
The parameter $\eta \in [0, 1]$ controls the scale of the stochastic noise added during the reverse generation steps:
- **$\eta = 0.0$**: The generation process is entirely deterministic. Starting from a fixed noise vector $z$, the model will always generate the exact same image.
- **$\eta = 1.0$**: The stochastic variance matches the DDPM transition probability, injecting maximum stochasticity at each step.

### 3. How does DDIM choose its smaller timestep subset from the original schedule?
DDIM selects a subsequence of timesteps $\tau_0, \tau_1, \dots, \tau_{S-1}$ from the original set $\{0, 1, \dots, T-1\}$. Typically, these are chosen using linear spacing (e.g., using `linspace(0, T - 1, S)`), meaning we skip equal intervals of noise steps (e.g. taking every 20th step for 50 steps from a 1000-step model).

### 4. What's the trade-off: when would you prefer DDPM over DDIM?
- **DDIM ($\eta=0$)** is preferred when sampling speed is critical (e.g. 50 steps instead of 1000) or when deterministic inversion / latent interpolation is required.
- **DDPM** (or DDIM with high $\eta$ and high step counts) can sometimes produce slightly better sample diversity and perceptual quality at the cost of significantly longer generation times, as the stochastic paths allow the model to correct errors made in early steps.

---

## Running the Benchmark
To run the benchmark script and generate the sample comparisons and timing chart on your machine, execute:
```bash
python benchmark.py
```
This script will produce the following deliverables in the `results/` directory:
- `ddpm_1000_steps.png`
- `ddim_10_steps.png`
- `ddim_25_steps.png`
- `ddim_50_steps.png`
- `ddim_100_steps.png`
- `timing_comparison.png`
- `latent_interpolation.png` (Bonus challenge demonstrating smooth transition between two noise vectors)
