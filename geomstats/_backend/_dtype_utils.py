import functools
import inspect
import types

from geomstats._backend import _backend_config as _config

_TO_UPDATE_FUNCS_DTYPE = []
_TO_UPDATE_FUNCS_KW_DTYPE = []


_MAP_FLOAT_TO_COMPLEX = {
    "float32": "complex64",
    "float64": "complex128",
}


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
        defaults[pos] = _config._DEFAULT_DTYPE
        func.__defaults__ = tuple(defaults)

    for func in _TO_UPDATE_FUNCS_KW_DTYPE:
        func.__kwdefaults__["dtype"] = _config._DEFAULT_DTYPE


def _modify_func_default_dtype(copy=True, kw_only=False, target=None):
    def _decorator(func):

        new_func = _copy_func(func) if copy else func

        if kw_only:
            _TO_UPDATE_FUNCS_KW_DTYPE.append(new_func)
        else:
            _TO_UPDATE_FUNCS_DTYPE.append(new_func)

        return new_func

    if target is None:
        return _decorator
    else:
        return _decorator(target)


def get_default_dtype():
    return _config._DEFAULT_DTYPE


def _dyn_update_dtype(dtype_pos=None, target=None):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            if dtype_pos is not None and len(args) > dtype_pos:
                args = list(args)
                args[dtype_pos] = _config._DEFAULT_DTYPE

            else:
                if kwargs.get("dtype", None) is None:
                    kwargs["dtype"] = _config._DEFAULT_DTYPE

            return func(*args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator
    else:
        return _decorator(target)


def _pre_set_default_dtype(as_dtype):
    def set_default_dtype(value):
        _config._DEFAULT_DTYPE = as_dtype(value)
        _config._DEFAULT_COMPLEX_DTYPE = as_dtype(_MAP_FLOAT_TO_COMPLEX[value])

        _update_default_dtypes()

        return get_default_dtype()

    return set_default_dtype


def _pre_cast_out_from_dtype(cast, is_floating, is_complex):
    def _cast_out_from_dtype(dtype_pos=None, target=None):
        def _decorator(func):
            @functools.wraps(func)
            def _wrapped(*args, **kwargs):
                out = func(*args, **kwargs)

                if is_floating(out) or is_complex(out):
                    if dtype_pos is not None and len(args) > dtype_pos:
                        dtype = args[dtype_pos]
                    else:
                        dtype = kwargs.get(
                            "dtype",
                            _config._DEFAULT_DTYPE
                            if is_floating(out)
                            else _config._DEFAULT_COMPLEX_DTYPE,
                        )

                    if out.dtype != dtype:
                        return cast(out, dtype)

                return out

            return _wrapped

        if target is None:
            return _decorator
        else:
            return _decorator(target)

    return _cast_out_from_dtype


def _pre_add_default_dtype(cast):
    def _add_default_dtype_by_casting(target=None):
        def _decorator(func):
            @functools.wraps(func)
            def _wrapped(*args, dtype=None, **kwargs):
                if dtype is None:
                    dtype = _config._DEFAULT_DTYPE

                out = func(*args, **kwargs)
                if out.dtype != dtype:
                    return cast(out, dtype)
                return out

            return _wrapped

        if target is None:
            return _decorator
        else:
            return _decorator(target)

    return _add_default_dtype_by_casting


def _pre_cast_fout_from_input_dtype(cast, is_floating):
    def _cast_fout_from_input_dtype(target=None):
        """Cast out of func if float and not accordingly to input.

        Notes
        -----
        Required for scipy when result is innacurate.
        """

        def _decorator(func):
            @functools.wraps(func)
            def _wrapped(x, *args, **kwargs):
                out = func(x, *args, **kwargs)
                if is_floating(x) and out.dtype != x.dtype:
                    return cast(out, x.dtype)
                return out

            return _wrapped

        if target is None:
            return _decorator
        else:
            return _decorator(target)

    return _cast_fout_from_input_dtype


def _np_box_unary_scalar(target=None):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x, *args, **kwargs):

            if type(x) is float:
                return func(x, *args, dtype=_config._DEFAULT_DTYPE, **kwargs)

            return func(x, *args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator
    else:
        return _decorator(target)


def _np_box_binary_scalar(target=None):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x1, x2, *args, **kwargs):

            if type(x1) is float:
                return func(x1, x2, *args, dtype=_config._DEFAULT_DTYPE, **kwargs)

            return func(x1, x2, *args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator
    else:
        return _decorator(target)
