import numpy as _np
import torch as _torch


def from_numpy(x):
    return _torch.from_numpy(x)


def array(val, dtype=None):
    if _torch.is_tensor(val):
        if dtype is None or val.dtype == dtype:
            return val.clone()

        return cast(val, dtype=dtype)

    if isinstance(val, _np.ndarray):
        tensor = from_numpy(val)
        if dtype is not None and tensor.dtype != dtype:
            tensor = cast(tensor, dtype=dtype)

        return tensor

    if isinstance(val, (list, tuple)) and len(val):
        tensors = [array(tensor, dtype=dtype) for tensor in val]
        return _torch.stack(tensors)

    return _torch.tensor(val, dtype=dtype)


def cast(x, dtype):
    if _torch.is_tensor(x):
        return x.to(dtype=dtype)
    return array(x, dtype=dtype)
