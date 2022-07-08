import functools
import inspect
import types

from numpy import dtype as _dtype

from ._common import cast

_DEFAULT_DTYPE = None


_TO_UPDATE_FUNCS_DTYPE = []


def _update_func_default_dtype(func):
    new_func = types.FunctionType(
        func.__code__,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        func.__closure__,
    )
    new_func.__dict__.update(func.__dict__)
    new_func.__kwdefaults__ = func.__kwdefaults__

    wrapped = getattr(func, "__wrapped__", None)
    if wrapped is not None:
        new_func.__wrapped__ = _update_func_default_dtype(wrapped)
        if "implementation" in new_func.__globals__:
            new_func.__globals__["implementation"] = new_func.__wrapped__
    else:
        _TO_UPDATE_FUNCS_DTYPE.append(new_func)
    return new_func


def _get_dtype_pos_in_defaults(func):
    pos = 0
    for name, parameter in inspect.signature(func).parameters.items():
        if name == "dtype":
            return pos
        if parameter.default is not inspect._empty:
            pos += 1
    else:
        raise Exception("dtype is not kwarg")


def _update_default_dtypes():
    funcs = _TO_UPDATE_FUNCS_DTYPE
    for func in funcs:
        pos = _get_dtype_pos_in_defaults(func)
        defaults = list(func.__defaults__)
        defaults[pos] = _DEFAULT_DTYPE
        func.__defaults__ = tuple(defaults)


def _add_dtype(func):
    @functools.wraps(func)
    def _wrapped(*args, dtype=None, **kwargs):
        if dtype is None:
            dtype = _DEFAULT_DTYPE

        out = func(*args, **kwargs)
        if out.dtype != dtype:
            return cast(out, dtype)
        return out

    return _wrapped


def _update_dtype(func):
    @functools.wraps(func)
    def _wrapped(*args, dtype=None, **kwargs):
        if dtype is None:
            dtype = _DEFAULT_DTYPE

        return func(*args, dtype=dtype, **kwargs)

    return _wrapped


def as_dtype(value):
    return _dtype(value)


def set_default_dtype(value):
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = as_dtype(value)
    _update_default_dtypes()

    return get_default_dtype()


def get_default_dtype():
    return _DEFAULT_DTYPE
