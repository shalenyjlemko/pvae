# Reproducing PVAE Experiments - Step by Step

This guide will help you reproduce the experiments from the paper "Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders" (NeurIPS 2019).

This is based on my experience getting the 2019 code working in 2025 with modern dependencies.

## Overview of Experiments

The paper includes three main experiments:
1. **Synthetic Tree Dataset** - Demonstrates hierarchical structure learning
2. **MNIST Dataset** - Compares Euclidean vs Poincaré embeddings
3. **Custom CSV Dataset** - For your own tabular data

---

## Experiment 1: Synthetic Tree Dataset

This experiment demonstrates how PVAE learns hierarchical tree structures in hyperbolic space. I found this to be the fastest experiment to run and understand the model's capabilities.

### 1a. Quick Test (Fast - 2 minutes)
```bash
conda activate pvae
cd <path_to_your_pvae_directory>

python pvae/main.py --model tree --manifold PoincareBall \
    --latent-dim 2 --hidden-dim 200 --prior-std 1.7 --c 1.2 \
    --data-size 50 --data-params 6 2 1 1 5 5 \
    --dec Wrapped --enc Wrapped \
    --prior RiemannianNormal --posterior RiemannianNormal \
    --epochs 10 --save-freq 10 --lr 1e-3 --batch-size 64 --iwae-samples 100
```

**What I observed:**
- Creates synthetic tree data with 6 branches, depth 2, specific parameters
- Trains a Poincaré VAE with 2D latent space
- Uses Riemannian Normal distribution (natural for hyperbolic space)
- Curvature c=1.2 (controls how "curved" the space is)

**Outputs saved to:** `experiments/<TIMESTAMP>/`
- `gen_samples_010.png` - Generated samples
- `reconstruct_010.png` - Reconstruction quality

### 1b. Full Paper Experiment (Slow - ~30 minutes)
```bash
python pvae/main.py --model tree --manifold PoincareBall \
    --latent-dim 2 --hidden-dim 200 --prior-std 1.7 --c 1.2 \
    --data-size 50 --data-params 6 2 1 1 5 5 \
    --dec Wrapped --enc Wrapped \
    --prior RiemannianNormal --posterior RiemannianNormal \
    --epochs 1000 --save-freq 100 --lr 1e-3 --batch-size 64 --iwae-samples 5000
```

**Key differences:**
- 1000 epochs (vs 10 for quick test)
- Saves every 100 epochs
- 5000 IWAE samples for better log-likelihood estimation

### 1c. Compare with Euclidean Baseline
```bash
python pvae/main.py --model tree --manifold Euclidean \
    --latent-dim 2 --hidden-dim 200 \
    --data-size 50 --data-params 6 2 1 1 5 5 \
    --dec Wrapped --enc Wrapped \
    --prior Normal --posterior Normal \
    --epochs 10 --save-freq 10 --lr 1e-3 --batch-size 64 --iwae-samples 100
```

**What I found:**
- Poincaré should show better hierarchy preservation
- Check `test_mlik` scores (higher is better)
- In my tests, Poincaré consistently outperformed Euclidean for hierarchical data

---

## Experiment 2: MNIST Dataset

This demonstrates embedding MNIST digits in 2D space. This was the most visually impressive experiment in my testing.

### 2a. Euclidean VAE (Baseline - ~20 minutes)
```bash
python pvae/main.py --model mnist --manifold Euclidean \
    --latent-dim 2 --hidden-dim 600 \
    --prior Normal --posterior Normal \
    --dec Wrapped --enc Wrapped \
    --lr 5e-4 --epochs 80 --save-freq 10 --batch-size 128 --iwae-samples 5000
```

**What to expect:**
- Downloads MNIST automatically to `data/` folder (first run only)
- Creates 2D embeddings of digits
- Generates reconstructions every 10 epochs
- Takes about 20 minutes on CPU

**Visualizations:**
- `gen_mean_XXX.png` - Generated digit from mean latent code
- `gen_means_XXX.png` - Grid of generated digits
- `recon_XXX.png` - Original vs reconstructed digits

### 2b. Poincaré VAE (~20 minutes)
```bash
python pvae/main.py --model mnist --manifold PoincareBall --c 0.7 \
    --latent-dim 2 --hidden-dim 600 \
    --prior WrappedNormal --posterior WrappedNormal \
    --dec Geo --enc Wrapped \
    --lr 5e-4 --epochs 80 --save-freq 10 --batch-size 128 --iwae-samples 5000
```

**What I observed:**
- Uses hyperbolic space (PoincareBall)
- Curvature c=0.7 (lower than tree experiment)
- `Geo` decoder (geodesic-based, works better for hyperbolic)
- WrappedNormal distribution

**My results:**
- Check final `test_mlik` (marginal log-likelihood)
- Look at the visual quality of reconstructions
- I found Poincaré gave better separation of digit classes in latent space

### 2c. Quick MNIST Test (Fast - 3 minutes)
```bash
python pvae/main.py --model mnist --manifold PoincareBall --c 0.7 \
    --latent-dim 2 --hidden-dim 600 \
    --prior WrappedNormal --posterior WrappedNormal \
    --dec Geo --enc Wrapped \
    --lr 5e-4 --epochs 5 --save-freq 5 --batch-size 128 --iwae-samples 100
```

---

## Experiment 3: Custom CSV Dataset

You can use your own tabular data with this model!

### 3a. Prepare Your Data

Create a CSV file in the `data/` folder of your pvae directory:
- **No header row**
- **Last column** = integer labels (class IDs)
- **All other columns** = features

Example: `data/mydata.csv`
```
0.5,0.2,0.8,1.0,0
0.3,0.4,0.6,0.9,0
0.1,0.9,0.2,0.3,1
...
```

### 3b. Run Experiment
```bash
python pvae/main.py --model csv --data-param mydata.csv \
    --data-size 4 --manifold PoincareBall --c 1.0 \
    --latent-dim 2 --hidden-dim 100 \
    --prior RiemannianNormal --posterior RiemannianNormal \
    --dec Wrapped --enc Wrapped \
    --epochs 100 --save-freq 10 --lr 1e-3 --batch-size 32
```

**Parameters:**
- `--data-param mydata.csv` - your CSV filename
- `--data-size 4` - number of features (columns - 1)

---

## Understanding the Outputs

### File Structure
```
experiments/
└── <TIMESTAMP><RANDOMID>/
    ├── gen_samples_XXX.png      # Generated samples
    ├── reconstruct_XXX.png       # Reconstructions
    ├── gen_mean_XXX.png          # (MNIST only) Mean generation
    ├── gen_means_XXX.png         # (MNIST only) Grid of generations
    ├── recon_XXX.png             # (MNIST only) Original vs recon
    ├── model.rar                 # Trained model weights
    └── losses.rar                # Training metrics
```

### Loading Saved Models

Here's how I analyzed trained models:
```python
import torch
from pvae import models

# Load model
model_path = 'experiments/<YOUR_RUN_ID>/model.rar'
model = torch.load(model_path)

# Load losses
losses = torch.load('experiments/<YOUR_RUN_ID>/losses.rar')
print(losses.keys())  # dict_keys(['train_loss', 'train_recon', 'train_kl', 'test_loss', 'test_mlik'])

# Plot training curves
import matplotlib.pyplot as plt
plt.plot(losses['train_loss'], label='Train Loss')
plt.plot(losses['test_loss'], label='Test Loss')
plt.legend()
plt.show()
```

---

## Visualizing the Latent Space

### For 2D Latent Spaces (What I Used)

I created this visualization script to explore the latent space:

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from pvae.models.mnist import Mnist
from pvae.models.tabular import Tree

# Load trained model
model = torch.load('experiments/<YOUR_RUN_ID>/model.rar')
model.eval()

# For MNIST: visualize digit embeddings
from torchvision import datasets, transforms
test_dataset = datasets.MNIST('data', train=False, download=True,
                               transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

latents = []
labels_list = []
with torch.no_grad():
    for data, labels in test_loader:
        qz_x = model.encode(data)
        z = qz_x.rsample()  # Sample from posterior
        latents.append(z.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

latents = np.vstack(latents)
labels_list = np.concatenate(labels_list)

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels_list, 
                     cmap='tab10', alpha=0.5, s=5)
plt.colorbar(scatter, label='Digit')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('MNIST Latent Space Embeddings')
plt.savefig('latent_space_visualization.png', dpi=300)
plt.show()
```

---

## Key Parameters Explained

### Manifold Parameters
- `--manifold PoincareBall` - Use hyperbolic geometry
- `--manifold Euclidean` - Use standard Euclidean space
- `--c [float]` - Curvature (higher = more curved, typical: 0.7-1.5)

### Architecture
- `--latent-dim 2` - Latent space dimensions (2 for visualization)
- `--hidden-dim 200` - Hidden layer size
- `--dec Wrapped` - Decoder with wrapped operations
- `--dec Geo` - Geodesic decoder (better for hyperbolic)
- `--dec Mob` - Möbius decoder
- `--enc Wrapped` - Wrapped encoder

### Distributions
- `--prior RiemannianNormal` - Natural for hyperbolic space
- `--prior WrappedNormal` - Wrapped normal distribution
- `--prior Normal` - Standard Gaussian (Euclidean)
- `--posterior [same options]` - For encoder distribution

### Training
- `--epochs 100` - Number of training epochs
- `--lr 1e-3` - Learning rate (1e-3 for synthetic, 5e-4 for MNIST)
- `--batch-size 64` - Batch size
- `--save-freq 10` - Save visualizations every N epochs
- `--iwae-samples 5000` - Samples for final log-likelihood (more = better but slower)

---

## Recommended Workflow

This is the approach I took when learning the codebase:

### Day 1: Quick Tests (What I Did First)
1. Run quick synthetic tree test (2 min)
2. Run quick MNIST test (3 min)
3. Review visualizations
4. Understand the outputs

### Day 2: Full Experiments
1. Run full synthetic tree experiment (30 min)
2. Run Euclidean baseline for comparison (30 min)
3. Analyze results

### Day 3: MNIST Deep Dive
1. Run full Euclidean MNIST (20 min)
2. Run full Poincaré MNIST (20 min)
3. Create latent space visualizations
4. Compare marginal log-likelihoods

### Day 4: Custom Data
1. Prepare your own dataset
2. Run experiments on custom data
3. Tune hyperparameters

---

## Troubleshooting

Issues I encountered and how I solved them:

### Out of Memory
- Reduce `--batch-size` (try 32 or 16)
- Reduce `--hidden-dim` (try 100 or 200)
- Reduce `--iwae-samples` for testing

### Training Not Converging
- Adjust learning rate (`--lr`)
- Try different curvature (`--c`)
- Increase epochs
- Check KL vs Reconstruction balance

### Visualizations Look Poor
- Train for more epochs
- Adjust architecture (decoder/encoder types)
- Try different prior std (`--prior-std`)

---

## Next Steps

What I plan to do next (and what you might consider):
1. Adapt the code for my sepsis project
2. Experiment with different architectures
3. Try different latent dimensions
4. Create custom visualizations
5. Compare metrics across configurations

Good luck with your experiments! 🚀
