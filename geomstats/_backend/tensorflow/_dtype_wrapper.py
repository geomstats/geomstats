import functools
import inspect
import types

import tensorflow as _tf
from tensorflow.dtypes import as_dtype

_DEFAULT_DTYPE = None

_TO_UPDATE_FUNCS_DTYPE = []
_TO_UPDATE_FUNCS_KW_DTYPE = []


# TODO: rename to target instead _func


def _copy_func(func):
    new_func = types.FunctionType(
        func.__code__,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        func.__closure__,
    )
    new_func.__dict__.update(func.__dict__)
    new_func.__kwdefaults__ = func.__kwdefaults__

    return new_func


def _update_func_default_dtype(copy=True, kw_only=False, _func=None):
    # TODO: modify_func_default_dtype instead?
    # TODO: rename also below

    def _decorator(func):

        new_func = _copy_func(func) if copy else func

        if kw_only:
            _TO_UPDATE_FUNCS_KW_DTYPE.append(new_func)
        else:
            _TO_UPDATE_FUNCS_DTYPE.append(new_func)

        return new_func

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


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
    for func in _TO_UPDATE_FUNCS_DTYPE:
        pos = _get_dtype_pos_in_defaults(func)
        defaults = list(func.__defaults__)
        defaults[pos] = _DEFAULT_DTYPE
        func.__defaults__ = tuple(defaults)

    for func in _TO_UPDATE_FUNCS_KW_DTYPE:
        func.__kwdefaults__["dtype"] = _DEFAULT_DTYPE


def _update_dtype(dtype_pos=None, _func=None):
    # TODO: rename to "update_dtype_dyn"?

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


def _cast_fout_from_dtype(dtype_pos=None, _func=None):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            out = func(*args, **kwargs)

            if out.dtype.is_floating:
                if dtype_pos is not None and len(args) > dtype_pos:
                    dtype = args[dtype_pos]
                else:
                    dtype = kwargs.get("dtype", _DEFAULT_DTYPE)

                if out.dtype != dtype:
                    return _tf.cast(out, dtype)

            return out

        return _wrapped

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def _box_unary_scalar(_func=None):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x, *args, **kwargs):

            if type(x) is float:
                x = _tf.constant(x, dtype=_DEFAULT_DTYPE)

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
                x1 = _tf.constant(x1, dtype=_DEFAULT_DTYPE)

            if type(x2) is float:
                x2 = _tf.constant(x2, dtype=_DEFAULT_DTYPE)

            return func(x1, x2, *args, **kwargs)

        return _wrapped

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def set_default_dtype(value):
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = as_dtype(value)
    _update_default_dtypes()

    return get_default_dtype()


def get_default_dtype():
    return _DEFAULT_DTYPE
