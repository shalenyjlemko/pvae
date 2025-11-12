# Poincaré VAE Setup Guide

This guide documents the steps taken to get the 6-year-old PVAE repository running on a modern system (November 2025).

## Repository Information
- **Original Repository**: https://github.com/emilemathieu/pvae
- **Paper**: Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders (NeurIPS 2019)
- **Local Path**: `C:\Users\denis\OneDrive\Desktop\Uni\M2\projects\vae_sepsis\pvae\pvae`

## Issues Encountered and Solutions

### 1. Missing Source Files
**Problem**: After cloning, the `pvae/` source directory appeared empty.
**Solution**: Files were tracked by git but deleted from working tree. Fixed with:
```bash
git restore .
```

### 2. Outdated Dependencies
**Original Requirements**:
- torch==1.3.1 (from 2019)
- numpy==1.22.0
- scipy==1.3.2
- scikit-learn==0.21.3
- Old geoopt commit from 2019

**Updated Requirements**:
- torch==2.4.1 (installed via conda)
- numpy==1.24.3
- scipy==1.10.1
- scikit-learn==1.3.2
- geoopt==0.5.1 (latest version)

### 3. Geoopt API Changes
**Problem**: The geoopt library underwent major API restructuring between 2019 and 2025.

**Error**:
```
ModuleNotFoundError: No module named 'geoopt.manifolds.poincare'
```

**Solution**: Modified `pvae/manifolds/poincareball.py` to add compatibility layer:

```python
# Compatibility fix for newer geoopt versions
try:
    from geoopt.manifolds.poincare.math import _lambda_x, arsinh, tanh
except (ImportError, ModuleNotFoundError):
    # For newer geoopt versions, use torch functions
    def _lambda_x(x, c, keepdim=False, dim=-1):
        """Conformal factor for Poincare ball"""
        return 2 / (1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim))
    
    arsinh = torch.asinh
    tanh = torch.tanh
```

## Setup Instructions

### Step 1: Clone the Repository
```bash
cd C:\Users\denis\OneDrive\Desktop\Uni\M2\projects\vae_sepsis
git clone https://github.com/emilemathieu/pvae.git pvae
cd pvae
git restore .  # Important: restore any missing files
```

### Step 2: Create Conda Environment
```bash
conda create -n pvae python=3.8 -y
conda activate pvae
```

### Step 3: Install PyTorch
```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

### Step 4: Install Additional Dependencies
```bash
pip install scikit-learn scipy seaborn matplotlib
```

### Step 5: Install Geoopt
```bash
pip install geoopt
```

### Step 6: Apply Compatibility Fix
The fix in `pvae/manifolds/poincareball.py` has already been applied. If starting fresh, modify the import section as shown above.

## Testing the Installation

### Test Import
```bash
python -c "import pvae; print('Success: pvae imported successfully')"
```

### Run Euclidean VAE (simpler baseline)
```bash
python pvae/main.py --model tree --manifold Euclidean \
    --latent-dim 2 --hidden-dim 200 --data-size 50 \
    --data-params 6 2 1 1 5 5 --dec Wrapped --enc Wrapped \
    --prior Normal --posterior Normal \
    --epochs 5 --save-freq 5 --lr 1e-3 --batch-size 64 --iwae-samples 100
```

### Run Poincaré VAE (main contribution)
```bash
python pvae/main.py --model tree --manifold PoincareBall \
    --latent-dim 2 --hidden-dim 200 --prior-std 1.7 --c 1.2 \
    --data-size 50 --data-params 6 2 1 1 5 5 \
    --dec Wrapped --enc Wrapped \
    --prior RiemannianNormal --posterior RiemannianNormal \
    --epochs 5 --save-freq 5 --lr 1e-3 --batch-size 64 --iwae-samples 100
```

## Running Full Experiments

### Synthetic Dataset (as in paper)
```bash
python pvae/main.py --model tree --manifold PoincareBall \
    --latent-dim 2 --hidden-dim 200 --prior-std 1.7 --c 1.2 \
    --data-size 50 --data-params 6 2 1 1 5 5 \
    --dec Wrapped --enc Wrapped \
    --prior RiemannianNormal --posterior RiemannianNormal \
    --epochs 1000 --save-freq 1000 --lr 1e-3 --batch-size 64 --iwae-samples 5000
```

### MNIST Dataset
```bash
# Euclidean VAE
python pvae/main.py --model mnist --manifold Euclidean \
    --latent-dim 2 --hidden-dim 600 --prior Normal --posterior Normal \
    --dec Wrapped --enc Wrapped \
    --lr 5e-4 --epochs 80 --save-freq 80 --batch-size 128 --iwae-samples 5000

# Poincaré VAE
python pvae/main.py --model mnist --manifold PoincareBall --c 0.7 \
    --latent-dim 2 --hidden-dim 600 --prior WrappedNormal --posterior WrappedNormal \
    --dec Geo --enc Wrapped \
    --lr 5e-4 --epochs 80 --save-freq 80 --batch-size 128 --iwae-samples 5000
```

## Key Files Modified
1. **`pvae/manifolds/poincareball.py`**: Added compatibility layer for geoopt API changes

## Output Location
Experiments are saved to: `experiments/[TIMESTAMP][RANDOMID]/`

## Known Warnings (Non-Critical)
You may see warnings about `arg_constraints` in custom distributions:
```
UserWarning: <class 'pvae.distributions.hyperbolic_radius.HyperbolicRadius'> does not define `arg_constraints`
```
These are cosmetic and do not affect functionality.

## Environment Summary
- **Python**: 3.8.20
- **PyTorch**: 2.4.1
- **Geoopt**: 0.5.1
- **NumPy**: 1.24.3
- **SciPy**: 1.10.1
- **Scikit-learn**: 1.3.2

## Quick Activation
To use the environment in the future:
```bash
conda activate pvae
cd C:\Users\denis\OneDrive\Desktop\Uni\M2\projects\vae_sepsis\pvae\pvae
```

## Troubleshooting

### Import Error: "ModuleNotFoundError: No module named 'pvae'"
Make sure you're in the correct directory:
```bash
cd C:\Users\denis\OneDrive\Desktop\Uni\M2\projects\vae_sepsis\pvae\pvae
```

### CUDA Errors
If you see CUDA errors, the code is trying to use GPU. The setup above uses CPU-only PyTorch. To force CPU:
```bash
export CUDA_VISIBLE_DEVICES=""  # Linux/Mac
$env:CUDA_VISIBLE_DEVICES=""    # PowerShell
```

### Slow Training
Reduce epochs and samples for testing:
- `--epochs 5` instead of 1000
- `--iwae-samples 100` instead of 5000

## Next Steps for Your Project
1. Explore the saved experiment outputs in `experiments/`
2. Read the paper to understand the model architecture
3. Adapt the code for your sepsis project
4. Consider creating custom datasets using the CSV loader
5. Experiment with different manifold parameters (--c for curvature)

## References
- Paper: https://arxiv.org/abs/1901.06033
- Original Repository: https://github.com/emilemathieu/pvae
- Geoopt Documentation: https://geoopt.readthedocs.io/
