# Diffusion Models from Scratch: Final Presentation Outline
**Course:** Seasons of Code 2026  
**Project:** Building a Class-Conditional Image Generator  

This document contains the slide-by-slide structure, content, and speaker notes for your 15-slide presentation.

---

## 📂 Slide Outline

### Section 1: Introduction (2 Slides)

#### Slide 1: Title Slide
* **Slide Title:** Diffusion Models from Scratch: Building Your Own Image Generator
* **Layout:** Clean, centered text with a dark mode aesthetic.
* **Content:**
  * **Presenter:** Arhan Khade
  * **Mentor:** Seasons of Code 2026
  * **Project:** Class-Conditional EMNIST Letters Generator
* **Speaker Notes:**
  > "Hello everyone, my name is Arhan, and today I will present my final project for the Seasons of Code program: 'Diffusion Models from Scratch'. Over the past 7 weeks, I built, trained, and optimized a class-conditional generative model from the ground up without using any black-box APIs."

#### Slide 2: Project Overview & Motivation
* **Slide Title:** Why Build Diffusion from Scratch?
* **Layout:** Two-column split: left column has bullet points; right column points to our final goal.
* **Content:**
  * **The Goal:** Demystify generative AI by implementing every component from first principles.
  * **MNIST to EMNIST:** Moving from basic digits to a larger, more complex dataset of handwritten characters.
  * **Core Components:** Custom UNet, Noise Scheduler, DDPM/DDIM Sampling loops, and Classifier-Free Guidance.
* **Speaker Notes:**
  > "Generative AI is everywhere, but it's often treated as a black box. The goal of this project was to understand the underlying mathematics and architecture. We started with the simple MNIST dataset and eventually graduated to the EMNIST letters dataset, which introduced real-world preprocessing challenges and required building robust data and conditioning pipelines."

---

### Section 2: Learnings (4 Slides)

#### Slide 3: PyTorch & Training Foundations
* **Slide Title:** Learnings: Building the ML Engine
* **Layout:** Bulleted list with key PyTorch APIs highlighted.
* **Content:**
  * **Autograd & Graph:** Deconstruct what happens during `loss.backward()` and how gradients flow.
  * **Custom Training Loops:** Bypassing high-level abstractions to write raw PyTorch optimization loops.
  * **Hardware Acceleration:** Managing device transfers (`.to(device)`) between CPU, MPS (Apple Silicon), and CUDA.
  * **Optimization Dynamics:** Observing convergence rate differences between SGD and Adam optimizers.
* **Speaker Notes:**
  > "In the first week, we focused on building PyTorch foundations. Understanding autograd, gradient accumulation, and device transfer was crucial. Writing training loops from scratch gave me direct control over optimization dynamics, highlighting why Adam outperforms SGD for complex loss landscapes."

#### Slide 4: The UNet Architecture
* **Slide Title:** Learnings: U-Net Convolutional Backbone
* **Layout:** Text on left, schematic representation of U-Net skips on right.
* **Content:**
  * **Feature Map Compression:** Downsampling to capture global context, upsampling to reconstruct details.
  * **Skip Connections:** Concatenating encoder features directly to the decoder.
  * **Why Skips Matter:** Prevents vanishing gradients and preserves spatial coordinates of fine details.
  * **Timestep Injection:** Injecting sinusoidal temporal embeddings into convolutional blocks.
* **Speaker Notes:**
  > "For Week 2, we built the U-Net architecture. U-Net is the backbone of diffusion models. Its key feature is skip connections, which bypass bottleneck layers to feed fine-grained spatial information from the encoder directly to the decoder. This prevents the blurry outputs common in simple CNN encoder-decoders."

#### Slide 5: The Forward Diffusion Process
* **Slide Title:** Learnings: Systematic Noise Scheduling
* **Layout:** Equation box on the left, schedule comparison plot on the right.
* **Content:**
  * **Closed-Form Sampling:** Sampling $x_t$ directly from $x_0$ using $\bar{\alpha}_t$ without iterative steps:
    $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
  * **Linear Schedule:** Constant step size, but details are destroyed too quickly at early steps.
  * **Cosine Schedule:** Keeps signal-to-noise ratios stable longer, improving detail retention.
* **Speaker Notes:**
  > "Week 3 introduced the forward process—systematically adding noise to an image. The mathematical breakthrough here is that we can jump directly to any timestep $t$ in closed form. We also compared linear and cosine noise schedules, learning that cosine schedules prevent the early, abrupt destruction of image structure."

#### Slide 6: The Reverse Process (DDPM)
* **Slide Title:** Learnings: Denoising and Generation
* **Layout:** Flow diagram showing: Noise -> UNet -> Predicted Noise -> Estimated Image.
* **Content:**
  * **Noise Prediction:** The UNet is trained to predict the added noise $\epsilon$, not the original image.
  * **Timestep Embeddings:** Conditioning the network using sinusoidal embeddings to indicate the noise level.
  * **Iterative Sampling:** Running Ho et al.'s Algorithm 2 to progressively remove noise over 1000 steps.
* **Speaker Notes:**
  > "In Week 4, we implemented the reverse process. Instead of asking the network to paint an image from scratch, we train it to predict the noise vector added at a specific timestep. During sampling, we use that prediction to subtract a fraction of the noise, repeating this iteratively to generate clean samples from pure noise."

---

### Section 3: Implementation and Final Results (6 Slides)

#### Slide 7: Speed Optimization: DDIM
* **Slide Title:** Implementation: Fast Sampling with DDIM
* **Layout:** Column layout comparing DDPM vs. DDIM.
* **Content:**
  * **The Bottleneck:** DDPM requires 1000 sequential passes through the UNet.
  * **DDIM (Deterministic):** Bypasses Markov chain constraints by utilizing a non-Markovian forward process.
  * **Performance:** Cut inference from 1000 steps to 25–50 steps with minimal impact on generation quality.
* **Speaker Notes:**
  > "Sampling with DDPM is slow because it requires 1000 sequential forward passes. In Week 5, I implemented DDIM sampling. DDIM is deterministic and allows us to skip steps, achieving comparable sample quality in only 25 to 50 steps, representing a 40x speedup."

#### Slide 8: Conditional Generation & CFG
* **Slide Title:** Implementation: Classifier-Free Guidance (CFG)
* **Layout:** Guidance formula highlighted.
* **Content:**
  * **Class Conditioning:** Injecting target label embeddings directly into the timestep embedding.
  * **Null Label Dropout:** Dropping label conditioning 15% of the time during training.
  * **Guidance Equation:**
    $$\epsilon_{cfg} = (1 + w)\epsilon_{cond} - w\epsilon_{uncond}$$
  * **Fidelity vs. Diversity:** Higher guidance scale $w$ yields sharper letters at the cost of style variety.
* **Speaker Notes:**
  > "Week 6 was about control: generating the specific letter we want. We implemented Classifier-Free Guidance. By dropping the class labels 15% of the time during training, the network learns to make both conditional and unconditional predictions. We then extrapolate between them using the guidance scale $w$."

#### Slide 9: Data Pipeline: EMNIST Letters
* **Slide Title:** Implementation: EMNIST Letters Pipeline
* **Layout:** Preprocessing flowchart showing: Load EMNIST -> Swap Axes (Transpose) -> Augment -> Scale.
* **Content:**
  * **Transposition Correction:** Fixed EMNIST's native flipped/rotated format.
  * **Label Mapping:** Mapped 1-indexed EMNIST labels (1–26) to 0-indexed classes (0–25).
  * **Augmentation Restrictions:** Rotations ($\pm 10^\circ$) and translations ($10\%$). Flipped images were strictly avoided to prevent semantic corruption.
* **Speaker Notes:**
  > "For Week 7, we trained on EMNIST Letters. This introduced data-engineering challenges. EMNIST is stored flipped and rotated, which we had to correct via a custom transposing transform. Furthermore, we had to carefully restrict our data augmentations: horizontal flips were disabled, because a flipped 'd' becomes a 'b', which would confuse the model."

#### Slide 10: Training Details & Logging
* **Slide Title:** Implementation: Training Run Specifications
* **Layout:** Left: Parameter Table. Right: Link to logs.
* **Content:**
  * **Hardware:** Run on a remote GPU machine (Conda superenv).
  * **Hyperparameters:**
    * Epochs: 25 | Batch Size: 128 | Learning Rate: 1e-4 (Adam)
    * Noise Steps: 1000 | Base Channels: 64 | Depth: 3
  * **Robust Logging:** Logs saved locally to [results/experiment_log.csv](./results/experiment_log.csv) and synced to Weights & Biases.
* **Speaker Notes:**
  > "The model was trained locally on a remote GPU server. We ran for 25 epochs using a batch size of 128 and an Adam optimizer at a learning rate of 1e-4. Training metrics and sample images were logged locally to a CSV file and synced to Weights & Biases."

#### Slide 11: Final Results Gallery
* **Slide Title:** Results: Generated Letter Grid
* **Layout:** Large centered display of the final generated letter grid.
* **Content:**
  * **Visual Reference:** [samples_epoch_25.png](./results/samples_epoch_25.png)
  * **Observations:** Model successfully captures distinct structural strokes for letters A–Z.
  * **Legibility:** Clean glyph forms with minor noise artifacts.
* **Speaker Notes:**
  > "This slide displays our generated samples at Epoch 25. The model successfully learned to write letters A through Z. The letters are legible, showing that class-conditioning successfully steers the generation toward distinct characters."

#### Slide 12: Guidance Sweep Analysis
* **Slide Title:** Results: Guidance Scale Analysis
* **Layout:** Showcase of letter generations at different guidance scales ($w = 0.0, 1.0, 3.0, 5.0$).
* **Content:**
  * **$w = 0.0$ (Unconditional):** Blurry, unguided shapes resembling random characters.
  * **$w = 1.0$:** Recognizable letters, but soft strokes and high style diversity.
  * **$w = 3.0$ (Sweet Spot):** Sharp, highly legible letters with structured margins.
  * **$w \ge 5.0$:** High contrast and thick lines; saturates features.
* **Speaker Notes:**
  > "Here, we analyze the impact of the guidance scale $w$. At $w=0$, the output is highly diverse but blurry. As we increase guidance to 3.0, the letters sharpen and align perfectly with their classes. At higher guidance values like 5.0, the lines thicken and become highly saturated, demonstrating the classic fidelity-diversity trade-off."

---

### Section 4: Future Implementation (2 Slides)

#### Slide 13: Scaling Architecture & Compute
* **Slide Title:** Future: Distributed Compute & Deeper Models
* **Layout:** Grid of two future improvements.
* **Content:**
  * **Distributed Training (DDP):** Implementing PyTorch Distributed Data Parallel to train across multi-GPU setups.
  * **Transformer Backbones (DiT):** Replacing the standard U-Net with a Diffusion Transformer (DiT) to scale parameters and capture long-range token relationships.
* **Speaker Notes:**
  > "Looking forward, the first path to improvement is scaling the model and compute. We can implement PyTorch's Distributed Data Parallel to scale training across multiple GPUs. Additionally, replacing the U-Net convolutional backbone with a Diffusion Transformer, or DiT, would allow the model to scale its parameters more efficiently."

#### Slide 14: Latent Diffusion & Text-to-Image
* **Slide Title:** Future: Latent Space & Text Prompting
* **Layout:** Comparative flow diagram of Pixel Space vs. Latent Space.
* **Content:**
  * **Latent Diffusion Models (LDM):** Train autoencoders to compress images into a latent space, conducting diffusion in low-resolution space (e.g. Stable Diffusion).
  * **Text Conditioning (CLIP):** Replace categorical embeddings with contextual text embeddings from a pre-trained CLIP model to enable true text-to-image prompts.
* **Speaker Notes:**
  > "The second path is architectural complexity. Pixel-space diffusion is computationally expensive for high-resolution images. Moving to Latent Diffusion, where diffusion is performed in a compressed latent space, would speed up training. We can also swap our simple class embeddings for CLIP text embeddings to enable descriptive text prompts."

---

### Section 5: Course Feedback (1 Slide)

#### Slide 15: Course Feedback
* **Slide Title:** Reflections & Course Feedback
* **Layout:** Three key reflection blocks.
* **Content:**
  * **Hands-on Learning:** Writing models and scheduling math from scratch is vastly superior to importing pre-existing libraries.
  * **Mentorship & Support:** The structured milestones and mentor checkpoints kept the project on track.
  * **Recommendation:** Keep the 'from scratch' philosophy—it builds a solid foundation for advanced deep learning topics.
* **Speaker Notes:**
  > "To wrap up, I want to thank the Seasons of Code mentors. Implementing a complex architecture like diffusion from scratch was challenging but incredibly rewarding. It demystified modern generative models and gave me deep confidence in building and debugging PyTorch systems. I highly recommend keeping this hands-on, first-principles approach for future cohorts. Thank you, and I am happy to take any questions."
