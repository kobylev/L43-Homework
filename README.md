# Denoising Autoencoder — MNIST

A convolutional denoising autoencoder trained on the MNIST dataset. The model learns to reconstruct clean digit images from corrupted inputs across a range of noise intensities.

---

## Project Schema & Data Flow

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PIPELINE                               │
│                                                                              │
│  MNIST                add_noise()             DenoisingAutoencoder           │
│  ┌──────────┐    ┌────────────────────┐    ┌───────────────────────────┐     │
│  │ Clean x  │───▶│ Random noise level │───▶│  Encoder                  │     │
│  │ [1×28×28]│    │ drawn natively     │    │  Conv 1×28×28→16×14×14    │     │
│  └──────────┘    │ from Gaussian      │    │  Conv 16×14×14→32×7×7     │     │
│                  └────────────────────┘    │  Conv 32×7×7→64×1×1       │     │
│                           │                └───────────┬───────────────┘     │
│                           │ Noisy x̃                    │ latent z            │
│                           │                ┌───────────▼───────────────┐     │
│                           │                │  Decoder                  │     │
│                           │                │  ConvT 64×1×1→32×7×7      │     │
│                           │                │  ConvT 32×7×7→16×14×14    │     │
│                           │                │  ConvT 16×14×14→1×28×28   │     │
│                           │                │  Sigmoid → x̂  [0,1]      │     │
│                           │                └───────────┬───────────────┘     │
│                           │                            │                     │
│  Loss = MSE(x̂, x_clean) ◀──────────────────────────────┘                    │
│  ▲ compared to CLEAN original, NOT the noisy input                           │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                             EVALUATION PIPELINE                              │
│                                                                              │
│  Test set (clean)  →  add fixed noise level  →  model(noisy)                 │
│  Noise swept: 0.00%, 0.50%, 1.00%, … 10.00%  (21 levels, step 0.5%)          │
│  Metrics per level: MSE mean/std · PSNR mean/std · SSIM mean/std             │
│  Outputs: evaluation_stats.csv · metrics_vs_noise.png                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```text
.
├── config.py                 # Hyperparameters and paths (single source of truth)
├── datasets.py               # Noise injection transform and DataLoaders
├── model.py                  # Encoder, Decoder, DenoisingAutoencoder class
├── train.py                  # Training loop, checkpoint save, loss curve plot
├── evaluate.py               # Metric computation, CSV export, all plots
├── main.py                   # Entry point — calls train() then evaluate()
├── interactive_noise.ipynb   # Interactive widget for demonstrating noise
├── requirements.txt          # Python dependencies
├── assets/
│   ├── training_curve.png          # MSE loss per epoch
│   ├── metrics_vs_noise.png        # MSE / PSNR / SSIM vs noise level
│   ├── sample_reconstructions.png  # Clean/Noisy/Recon visual grid
│   └── evaluation_stats.csv        # Full metrics table across sweeps
├── models/
│   └── denoising_autoencoder.pth   # Saved model weights
└── data/                           # Auto-downloaded MNIST
```

---

## Architecture

### Encoder

Progressively maps spatial dimensions downwards while increasing channels, compressing the image into a deep latent spatial bottleneck.

| Layer | Input | Output | Operation |
|-------|-------|--------|-----------|
| Conv2d + ReLU | 1×28×28 | 16×14×14 | kernel 3, stride 2, pad 1 |
| Conv2d + ReLU | 16×14×14 | 32×7×7 | kernel 3, stride 2, pad 1 |
| Conv2d + ReLU | 32×7×7 | 64×1×1 | kernel 7, stride 1, pad 0 |

### Decoder

Mirrors the encoder with transposed convolutions to restore the original 28×28 spatial resolution smoothly.

| Layer | Input | Output | Operation |
|-------|-------|--------|-----------|
| ConvTranspose2d + ReLU | 64×1×1 | 32×7×7 | kernel 7, stride 1, pad 0 |
| ConvTranspose2d + ReLU | 32×7×7 | 16×14×14 | kernel 3, stride 2, pad 1, out_pad 1 |
| ConvTranspose2d + **Sigmoid** | 16×14×14 | **1×28×28** | kernel 3, stride 2, pad 1, out_pad 1 |

### Loss function

```text
Loss = MSE(x̂, x_clean)
```

The loss is computed against the **original clean image**, not the noisy input. This forces the model to learn the structural integrity of the digits rather than memorising the static overlay.

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate        
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train and evaluate

```bash
python main.py
```

The script will:
1. Download MNIST automatically into `./data/`
2. Train for 10 epochs on GPU if available
3. Save the model to `./models/denoising_autoencoder.pth`
4. Evaluate across 21 noise levels (0.00 % → 10.00 % in 0.5 % steps)
5. Write all plots and the CSV to `./assets/`

### Run only training or evaluation separately

```bash
python train.py      # train and save weights
python evaluate.py   # load saved weights and evaluate
```

### Key hyperparameters (edit `config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 10 | Training epochs |
| `BATCH_SIZE` | 128 | Batch size |
| `LEARNING_RATE` | 1e-3 | Adam learning rate |
| `TRAIN_NOISE_FACTOR` | 0.1 | Default train noise magnitude (10%) |

---

## Results & Analysis

### Training loss

![Training Loss](assets/training_curve.png)

The loss followed a very steep exponential decay within the first 2 epochs, plummeting as the model rapidly mapped out the core outlines of the numeric digits. Afterward, it settled into a gentle curve reaching a stable minimum MSE of around **0.003**.

---

### Reconstruction quality vs. noise level

![Metrics vs Noise](assets/metrics_vs_noise.png)

**MSE (lower = better)**
The reconstruction error rises smoothly but stays below 0.005 even at the maximum testing extremity of 10% noise. The variance is consistent, demonstrating the autoencoder behaves symmetrically across most digit classes.

**PSNR (higher = better)**
Peak Signal-to-Noise Ratio starts high (nearly 30 dB) when practically zero noise is applied, meaning the autoencoder is an excellent straight passthrough mechanism. At 10% severe distortion, it drops toward roughly 23-24 dB, which is still comfortably above the 20 dB accepted visual quality threshold.

**SSIM (higher = better, max = 1)**
Structural Similarity tracks brilliantly. Even near zero noise, it scores well above 0.95. At the 10% extreme tested parameter, it only deteriorates to approximately 0.70. Since SSIM aggressively punishes blurriness—which MSE loss inherently creates on noisy edges—remaining above 0.7 at 10% corruption proves the model retained the categorical edge bounds of the digits securely.

---

### Visual reconstructions

![Sample Reconstructions](assets/sample_reconstructions.png)

Each row sweeps from Clean (0%) to Noisy (various bounds) to the Reconstructed final result. The network clears out the scattered static beautifully while resisting heavy blurring on the structural outlines of the original numbers. 

---

## Output files

| File | Description |
|------|-------------|
| `models/denoising_autoencoder.pth` | PyTorch model weights |
| `assets/training_curve.png` | MSE loss curve over epochs |
| `assets/metrics_vs_noise.png` | MSE / PSNR / SSIM vs noise level (0–10 %) |
| `assets/sample_reconstructions.png` | Visual Clean/Noisy/Recon grid |
| `assets/evaluation_stats.csv` | Per-level mean and std for all three metrics |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` + `torchvision` | Model, training, dataset |
| `numpy` | Numerical operations |
| `pandas` | Sweeping Metrics CSV exporting |
| `matplotlib` | Plotting |
| `scikit-image` | PSNR and SSIM metrics |
| `tqdm` | Progress bars |
| `ipywidgets` | Jupyter Notebook interactive UI components |

---

## Dataset

**MNIST** — handwritten digits dataset formulated by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.

The dataset consists of 70,000 grayscale 28×28 images across 10 digit categories (0 through 9). The dataset is downloaded automatically by `torchvision.datasets.MNIST` on first run and cached in `./data/`.
