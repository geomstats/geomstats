import functools
import inspect
import types

import numpy as _np

from ._common import cast

_DEFAULT_DTYPE = None
_DEFAULT_COMPLEX_DTYPE = None


_TO_UPDATE_FUNCS_DTYPE = []


_MAP_FLOAT_TO_COMPLEX = {
    "float32": "complex64",
    "float64": "complex128",
}


def _update_func_default_dtype(func):
    # TODO: copy and update
    new_func = types.FunctionType(
        func.__code__,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        func.__closure__,
    )
    new_func.__dict__.update(func.__dict__)
    new_func.__kwdefaults__ = func.__kwdefaults__

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


def _add_default_dtype(func):
    @functools.wraps(func)
    def _wrapped(*args, dtype=None, **kwargs):
        if dtype is None:
            dtype = _DEFAULT_DTYPE

        out = func(*args, **kwargs)
        if out.dtype != dtype:
            return cast(out, dtype)
        return out

    return _wrapped


def _update_dtype(dtype_pos=None, _func=None):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            if dtype_pos is not None and len(args) > dtype_pos:
                args = list(args)
                args[dtype_pos] = _DEFAULT_DTYPE

            else:
                if kwargs.get("dtype", None) is None:
                    kwargs["dtype"] = _DEFAULT_DTYPE

            return func(*args, **kwargs)

        return _wrapped

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def _cast_out_from_dtype(dtype_pos=None, _func=None):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            out = func(*args, **kwargs)

            if out.dtype.kind == "f" or out.dtype.kind == "c":
                if dtype_pos is not None and len(args) > dtype_pos:
                    dtype = args[dtype_pos]
                else:
                    dtype = kwargs.get(
                        "dtype",
                        _DEFAULT_DTYPE
                        if out.dtype.kind == "f"
                        else _DEFAULT_COMPLEX_DTYPE,
                    )

                if out.dtype != dtype:
                    return cast(out, dtype)

            return out

        return _wrapped

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def _cast_fout_from_input_dtype(func):
    """Cast out of func if float and not accordingly to input.

    Notes
    -----
    Required for scipy when result is innacurate.
    """

    @functools.wraps(func)
    def _wrapped(x, *args, **kwargs):
        out = func(x, *args, **kwargs)
        if out.dtype.kind == "f" and out.dtype != x.dtype:
            return cast(out, x.dtype)
        return out

    return _wrapped


def _box_unary_scalar(_func=None):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x, *args, **kwargs):

            if type(x) is float:
                return func(x, *args, dtype=_DEFAULT_DTYPE, **kwargs)

            return func(x, *args, **kwargs)

        return _wrapped

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def _box_binary_scalar(_func=None):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x1, x2, *args, **kwargs):

            if type(x1) is float:
                return func(x1, x2, *args, dtype=_DEFAULT_DTYPE, **kwargs)

            return func(x1, x2, *args, **kwargs)

        return _wrapped

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def as_dtype(value):
    return _np.dtype(value)


def set_default_dtype(value):
    global _DEFAULT_DTYPE
    global _DEFAULT_COMPLEX_DTYPE

    _DEFAULT_DTYPE = as_dtype(value)
    _DEFAULT_COMPLEX_DTYPE = as_dtype(_MAP_FLOAT_TO_COMPLEX[value])
    _update_default_dtypes()

    return get_default_dtype()


def get_default_dtype():
    return _DEFAULT_DTYPE
