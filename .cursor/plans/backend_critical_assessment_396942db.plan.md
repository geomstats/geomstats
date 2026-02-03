---
name: Backend Array API Migration
overview: Migrate geomstats backend to use Python Array API standard via numpy 2.0, eliminating code duplication while preserving backward compatibility.
todos:
  - id: array-api-setup
    content: "Phase 1: Bump numpy to 2.0, create _array_api.py with get_namespace() helper"
    status: completed
  - id: migrate-math-ops
    content: "Phase 2a: Migrate math functions (abs, sin, cos, exp, log, etc.) to Array API"
    status: completed
  - id: migrate-array-creation
    content: "Phase 2b: Migrate array creation functions (array, zeros, ones, eye, etc.) to Array API"
    status: completed
  - id: migrate-reductions
    content: "Phase 2c: Migrate reduction functions (sum, mean, prod, min, max, etc.) to Array API"
    status: completed
  - id: test-hypersphere
    content: "Phase 3a: Run hypersphere tests on all three backends"
    status: completed
  - id: test-expand
    content: "Phase 3b: Expand tests to more geometry modules"
    status: completed
isProject: false
---

# Array API Migration Plan for Geomstats Backend

## Goal

Replace ~80% of duplicated backend code with Python Array API standard calls, using numpy 2.0's native Array API support. This preserves the existing `GEOMSTATS_BACKEND` environment variable while eliminating code duplication.

## Design Decisions

Based on clarification:

- **Bump numpy to >= 2.0** (simpler, Array API native support)
- **Preserve env variable behavior** (backward compatible, gradual migration)
- **Keep scipy fallbacks for now** (focus on core Array API first)

---

## Phase 1: Setup Array API Infrastructure

### 1.1 Update Dependencies

Update [`pyproject.toml`](pyproject.toml):

```python
# Change line 33 from:
"numpy >= 1.18.1",
# To:
"numpy >= 2.0",
```

### 1.2 Create Array API Helper Module

Create new file [`geomstats/_backend/_array_api.py`](geomstats/_backend/_array_api.py):

```python
"""Array API standard utilities for backend-agnostic operations.

This module provides helpers to use Python Array API standard functions
that work identically across numpy, pytorch, and other compatible backends.
"""

import numpy as np

# For PyTorch compatibility with Array API
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def get_namespace(*arrays):
    """Get the array namespace (xp) for the given arrays.
    
    Uses numpy 2.0's __array_namespace__ protocol.
    Falls back to numpy for Python scalars/lists.
    
    Parameters
    ----------
    *arrays : array-like
        Input arrays to determine namespace from.
    
    Returns
    -------
    xp : module
        Array namespace (numpy, torch, etc.)
    """
    for arr in arrays:
        if hasattr(arr, '__array_namespace__'):
            return arr.__array_namespace__()
        if _HAS_TORCH and isinstance(arr, torch.Tensor):
            return torch
    return np


def is_torch_tensor(x):
    """Check if x is a PyTorch tensor."""
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def is_array_api_compatible(x):
    """Check if x supports the Array API standard."""
    return hasattr(x, '__array_namespace__')
```

### 1.3 Key Insight: PyTorch Array API Compatibility

PyTorch 2.0+ supports Array API. Enable it with:

```python
# In pytorch/__init__.py, add near top:
import torch
torch.set_default_device('cpu')  # Ensures consistent behavior
```

PyTorch's Array API can be accessed via `torch.__array_namespace__()` or by using torch functions directly (they're mostly compatible).

---

## Phase 2: Migrate Functions to Array API

### Functions to Migrate (categorized)

#### 2a. Math Functions (~25 functions)

These are directly available in Array API:

| Current | Array API |

|---------|-----------|

| `abs` | `xp.abs` |

| `sin`, `cos`, `tan` | `xp.sin`, `xp.cos`, `xp.tan` |

| `arcsin`, `arccos`, `arctan2` | `xp.asin`, `xp.acos`, `xp.atan2` |

| `sinh`, `cosh`, `tanh` | `xp.sinh`, `xp.cosh`, `xp.tanh` |

| `exp`, `log`, `sqrt` | `xp.exp`, `xp.log`, `xp.sqrt` |

| `ceil`, `floor` | `xp.ceil`, `xp.floor` |

| `sign` | `xp.sign` |

| `real`, `imag`, `conj` | `xp.real`, `xp.imag`, `xp.conj` |

**Migration pattern** (in each backend's `__init__.py`):

```python
# OLD (pytorch/__init__.py)
abs = _box_unary_scalar(target=_torch.abs)
sin = _box_unary_scalar(target=_torch.sin)

# NEW (shared, works for all backends)
from ._array_api import get_namespace

def abs(x):
    xp = get_namespace(x)
    return xp.abs(x)

def sin(x):
    xp = get_namespace(x)
    return xp.sin(x)
```

**Note**: Array API uses `asin`/`acos` not `arcsin`/`arccos`. We need wrapper functions:

```python
def arcsin(x):
    xp = get_namespace(x)
    return xp.asin(x)

def arccos(x):
    xp = get_namespace(x)
    return xp.acos(x)
```

#### 2b. Array Creation Functions (~15 functions)

| Current | Array API |

|---------|-----------|

| `array` | `xp.asarray` |

| `zeros`, `ones` | `xp.zeros`, `xp.ones` |

| `zeros_like`, `ones_like` | `xp.zeros_like`, `xp.ones_like` |

| `empty`, `empty_like` | `xp.empty`, `xp.empty_like` |

| `eye` | `xp.eye` |

| `arange` | `xp.arange` |

| `linspace` | `xp.linspace` |

**Migration example**:

```python
def zeros(shape, dtype=None):
    # Use default dtype from config if not specified
    if dtype is None:
        dtype = _config.DEFAULT_DTYPE
    xp = np  # Array creation uses numpy by default
    return xp.zeros(shape, dtype=dtype)

def zeros_like(x, dtype=None):
    xp = get_namespace(x)
    return xp.zeros_like(x, dtype=dtype)
```

#### 2c. Reduction Functions (~15 functions)

| Current | Array API |

|---------|-----------|

| `sum` | `xp.sum` |

| `mean` | `xp.mean` |

| `prod` | `xp.prod` |

| `min`, `max` | `xp.min`, `xp.max` |

| `all`, `any` | `xp.all`, `xp.any` |

| `std` | `xp.std` |

**Key difference**: Array API uses `axis` not `dim` (like numpy):

```python
def sum(x, axis=None, keepdims=False, dtype=None):
    xp = get_namespace(x)
    return xp.sum(x, axis=axis, keepdims=keepdims, dtype=dtype)
```

#### 2d. Shape/Index Functions (~20 functions)

| Current | Array API |

|---------|-----------|

| `reshape` | `xp.reshape` |

| `transpose` | `xp.permute_dims` (note: different name!) |

| `squeeze`, `expand_dims` | `xp.squeeze`, `xp.expand_dims` |

| `concatenate`, `stack` | `xp.concat`, `xp.stack` |

| `where` | `xp.where` |

**Important**: Some names differ:

- `transpose` → `permute_dims`
- `concatenate` → `concat`
```python
def transpose(x, axes=None):
    xp = get_namespace(x)
    if axes is None:
        axes = tuple(range(x.ndim - 1, -1, -1))
    return xp.permute_dims(x, axes)

def concatenate(arrays, axis=0):
    xp = get_namespace(*arrays)
    return xp.concat(arrays, axis=axis)
```


### Functions to Keep Backend-Specific (~20%)

These are NOT in Array API standard and need backend-specific code:

1. **Autodiff** (`gs.autodiff.*`): Completely backend-specific
2. **Advanced linalg** (`gs.linalg.expm`, `logm`, `sqrtm`, `solve_sylvester`): Use scipy
3. **Custom operations**: `einsum`, `assignment`, `scatter_add`, `mat_from_diag_triu_tril`
4. **Scipy special functions**: `erf`, `gamma`, `polygamma`

---

## Phase 3: Testing Strategy

### 3a. Test Hypersphere First

Run hypersphere tests on each backend:

```bash
# NumPy backend
GEOMSTATS_BACKEND=numpy pytest tests/tests_geomstats/test_geometry/test_hypersphere.py -v

# PyTorch backend  
GEOMSTATS_BACKEND=pytorch pytest tests/tests_geomstats/test_geometry/test_hypersphere.py -v

# Autograd backend
GEOMSTATS_BACKEND=autograd pytest tests/tests_geomstats/test_geometry/test_hypersphere.py -v
```

**Why hypersphere first?**

- Uses core array operations (sin, cos, einsum, matmul)
- Well-tested, stable module
- Exercises both extrinsic and intrinsic coordinates
- Tests Riemannian metric operations

### 3b. Expand to More Tests

After hypersphere passes, expand in this order:

1. **Core geometry** (low complexity):
   ```bash
   pytest tests/tests_geomstats/test_geometry/test_euclidean.py
   pytest tests/tests_geomstats/test_geometry/test_matrices.py
   ```

2. **Lie groups** (medium complexity):
   ```bash
   pytest tests/tests_geomstats/test_geometry/test_special_orthogonal.py
   pytest tests/tests_geomstats/test_geometry/test_special_euclidean.py
   ```

3. **Full geometry suite**:
   ```bash
   pytest tests/tests_geomstats/test_geometry/ -v --ignore=tests/tests_geomstats/test_geometry/test_discrete_curves.py
   ```

4. **Full test suite** (only after above pass):
   ```bash
   pytest tests/tests_geomstats/ -m "not (slow or redundant)"
   ```


---

## Implementation Order

### Step 1: Create `_array_api.py` (new file)

Location: `geomstats/_backend/_array_api.py`

Contents: `get_namespace()`, `is_torch_tensor()`, helper functions

### Step 2: Update `pyproject.toml`

Bump numpy version to >= 2.0

### Step 3: Migrate numpy backend first

Update `geomstats/_backend/numpy/__init__.py`:

- Import from `_array_api`
- Replace direct numpy imports with Array API wrappers
- Test with `GEOMSTATS_BACKEND=numpy pytest test_hypersphere.py`

### Step 4: Migrate pytorch backend

Update `geomstats/_backend/pytorch/__init__.py`:

- Remove `_box_unary_scalar` / `_box_binary_scalar` wrappers for migrated functions
- Use Array API versions
- Test with `GEOMSTATS_BACKEND=pytorch pytest test_hypersphere.py`

### Step 5: Migrate autograd backend

Update `geomstats/_backend/autograd/__init__.py`:

- Similar to numpy (autograd.numpy is Array API compatible)
- Test with `GEOMSTATS_BACKEND=autograd pytest test_hypersphere.py`

### Step 6: Run expanded tests

After all three backends pass hypersphere, expand testing progressively.

---

## Risk Mitigation

1. **Incremental migration**: Migrate one category at a time, test after each
2. **Keep old code commented**: Don't delete until tests pass
3. **Backend-specific fallbacks**: If Array API version fails, fall back to original
```python
def sin(x):
    try:
        xp = get_namespace(x)
        return xp.sin(x)
    except Exception:
        # Fallback to backend-specific
        if is_torch_tensor(x):
            return torch.sin(x)
        return np.sin(x)
```


---

## Expected Outcome

After this migration:

- **~80% of functions** use shared Array API code
- **~20% remain** backend-specific (autodiff, advanced linalg, custom ops)
- **Code reduction**: ~1500 lines removed from combined backends
- **No breaking changes** to user API (`import geomstats.backend as gs` works identically)