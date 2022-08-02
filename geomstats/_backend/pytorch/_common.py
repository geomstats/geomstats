import numpy as _np
import torch as _torch


def from_numpy(x, dtype=None):
    tensor = _torch.from_numpy(x)

    if dtype is not None and tensor.dtype != dtype:
        tensor = cast(tensor, dtype=dtype)

    return tensor


def array(val, dtype=None):
    if _torch.is_tensor(val):
        if dtype is None or val.dtype == dtype:
            return val.clone()
        else:
            return cast(val, dtype=dtype)

    elif isinstance(val, _np.ndarray):
        return from_numpy(val, dtype=dtype)

    elif isinstance(val, (list, tuple)) and len(val):
        tensors = [array(tensor, dtype=dtype) for tensor in val]
        return _torch.stack(tensors)

    return _torch.tensor(val, dtype=dtype)


def cast(x, dtype):
    if _torch.is_tensor(x):
        return x.to(dtype=dtype)
    return array(x, dtype=dtype)
