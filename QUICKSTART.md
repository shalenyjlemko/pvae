# Quick Start Commands

## First Time Setup (For Colleagues)

If you're setting up the environment for the first time, follow these steps:

### 1. Navigate to the Repository
```powershell
cd C:\Users\denis\OneDrive\Desktop\Uni\M2\projects\vae_sepsis\pvae\pvae
```

### 2. Create Conda Environment
```powershell
conda create -n pvae python=3.8 -y
```

### 3. Activate Environment
```powershell
conda activate pvae
```

### 4. Install PyTorch
```powershell
conda install pytorch torchvision cpuonly -c pytorch -y
```

### 5. Install Additional Dependencies
```powershell
pip install scikit-learn scipy seaborn matplotlib
```

### 6. Install Geoopt (Hyperbolic Geometry Library)
```powershell
pip install geoopt
```

### 7. Test Installation
```powershell
python -c "import pvae; print('Success! PVAE is ready to use.')"
```

---

## For Existing Environment (Daily Use)

### Activate Environment
```powershell
conda activate pvae
cd C:\Users\denis\OneDrive\Desktop\Uni\M2\projects\vae_sepsis\pvae\pvae
```

## Quick Test (30 seconds)
```bash
python pvae/main.py --model tree --manifold PoincareBall --latent-dim 2 --hidden-dim 100 --prior-std 1.7 --c 1.2 --data-size 50 --data-params 6 2 1 1 5 5 --dec Wrapped --enc Wrapped --prior RiemannianNormal --posterior RiemannianNormal --epochs 3 --save-freq 3 --lr 1e-3 --batch-size 32 --iwae-samples 50
```

## Experiment from Paper (slower)
```bash
python pvae/main.py --model tree --manifold PoincareBall --latent-dim 2 --hidden-dim 200 --prior-std 1.7 --c 1.2 --data-size 50 --data-params 6 2 1 1 5 5 --dec Wrapped --enc Wrapped --prior RiemannianNormal --posterior RiemannianNormal --epochs 1000 --save-freq 100 --lr 1e-3 --batch-size 64 --iwae-samples 5000
```

## View Results
Results are saved in: `experiments/[TIMESTAMP]/`
