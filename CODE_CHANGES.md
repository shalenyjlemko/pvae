# Code Changes for 2025 Compatibility

## Summary
Only **one file** needed modification to make the 6-year-old code work with modern dependencies.

## Changed File
**File**: `pvae/manifolds/poincareball.py`

### Original Code (lines 1-5)
```python
import torch
from geoopt.manifolds import PoincareBall as PoincareBallParent
from geoopt.manifolds.poincare.math import _lambda_x, arsinh, tanh

MIN_NORM = 1e-15
```

### Updated Code
```python
import torch
from geoopt.manifolds import PoincareBall as PoincareBallParent

# Compatibility fix for newer geoopt versions
# In newer versions, math functions are methods of the manifold
try:
    from geoopt.manifolds.poincare.math import _lambda_x, arsinh, tanh
except (ImportError, ModuleNotFoundError):
    # For newer geoopt versions, we'll use torch functions and manifold methods
    def _lambda_x(x, c, keepdim=False, dim=-1):
        """Conformal factor for Poincare ball"""
        return 2 / (1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim))
    
    arsinh = torch.asinh
    tanh = torch.tanh

MIN_NORM = 1e-15
```

## Explanation
The `geoopt` library reorganized its API between version 0.0.1 (2019) and 0.5.1 (2025):

1. **Old structure**: Math functions lived in `geoopt.manifolds.poincare.math`
2. **New structure**: Math functions are now methods of the manifold classes

The compatibility fix:
- First tries to import from the old location (for backward compatibility)
- If that fails, defines the needed functions using standard PyTorch operations
- `_lambda_x`: Conformal factor calculation for Poincaré ball
- `arsinh`: Maps to PyTorch's `torch.asinh`
- `tanh`: Maps to PyTorch's `torch.tanh`

## No Other Changes Required
All other compatibility issues were resolved by:
- Using modern package versions (PyTorch 2.4.1, geoopt 0.5.1)
- Using Python 3.8 (maintains compatibility with both old and new packages)

## Testing
Both test runs completed successfully:
- ✅ Euclidean VAE baseline
- ✅ Poincaré VAE (main contribution)

## Future-Proofing
This change makes the code compatible with both old and new geoopt versions, so it should continue working even if geoopt updates further.
