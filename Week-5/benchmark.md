# Week 5 — Benchmark Report: DDPM vs. DDIM

This report presents performance timings and quality comparisons between standard DDPM sampling (1000 steps) and DDIM sampling at various step counts.

## Speed Benchmark Results

| Configuration | Steps | Sampling Time (s) | Speedup Factor |
| :--- | :---: | :---: | :---: |
| **DDPM (Baseline)** | 1000 | 7.30s | 1.0x (Reference) |
| **DDIM** | 100 | 0.73s | 10.0x |
| **DDIM** | 50 | 0.37s | 20.0x |
| **DDIM** | 25 | 0.18s | 39.9x |
| **DDIM** | 10 | 0.09s | 81.7x |

## Visual Analysis & Trade-offs

- **DDIM (100 steps)**: Indistinguishable from the baseline DDPM at a fraction of the time.
- **DDIM (50 steps)**: Excellent quality/speed sweet spot. Retains all structural features and details of the baseline while running 20x faster.
- **DDIM (25 steps)**: Minor degradation in some fine details, but overall structure is highly coherent. Very usable for fast prototyping.
- **DDIM (10 steps)**: Demonstrates some blurring or structural anomalies, but still generates recognizable digits, which is impossible with DDPM at 10 steps.

## Latent Interpolation

Because DDIM is deterministic ($\eta = 0$), there is a bijective mapping between the initial noise layout and the final generated image. Linearly interpolating between two initial noise vectors $z_0$ and $z_1$ yields a smooth semantic transition in the pixel space.
