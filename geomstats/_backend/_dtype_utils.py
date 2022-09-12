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
    """Copy function."""
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
    """Get dtype position in defaults."""
    pos = 0
    for name, parameter in inspect.signature(func).parameters.items():
        if name == "dtype":
            return pos
        if parameter.default is not inspect._empty:
            pos += 1

    raise Exception("dtype is not kwarg")


def _update_default_dtypes():
    """Update default dtype of functions.

    Notice it (mutably) changes function defaults. For external functions,
    copy the functions first to avoid surprising users.
    """
    for func in _TO_UPDATE_FUNCS_DTYPE:
        pos = _get_dtype_pos_in_defaults(func)
        defaults = list(func.__defaults__)
        defaults[pos] = _config.DEFAULT_DTYPE
        func.__defaults__ = tuple(defaults)

    for func in _TO_UPDATE_FUNCS_KW_DTYPE:
        func.__kwdefaults__["dtype"] = _config.DEFAULT_DTYPE


def _modify_func_default_dtype(copy=True, kw_only=False, target=None):
    """Modify function default dtype by acting directly in the object.

    Parameters
    ----------
    copy: bool
        If true, copies function before changing dtype.
    kw_only : bool
        If true, it is assumed dtype is kwarg only argument.
    """

    def _decorator(func):

        new_func = _copy_func(func) if copy else func

        if kw_only:
            _TO_UPDATE_FUNCS_KW_DTYPE.append(new_func)
        else:
            _TO_UPDATE_FUNCS_DTYPE.append(new_func)

        return new_func

    if target is None:
        return _decorator

    return _decorator(target)


def get_default_dtype():
    """Get backend default float dtype."""
    return _config.DEFAULT_DTYPE


def _dyn_update_dtype(dtype_pos=None, target=None):
    """Update (dynamically) function dtype.

    When function is called, it verifies if dtype is passed. If not, default
    dtype is set.
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            if dtype_pos is not None and len(args) > dtype_pos:
                args = list(args)
                args[dtype_pos] = _config.DEFAULT_DTYPE

            else:
                if kwargs.get("dtype") is None:
                    kwargs["dtype"] = _config.DEFAULT_DTYPE

            return func(*args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)


def _pre_set_default_dtype(as_dtype):
    def set_default_dtype(value):
        """Set backend default dtype."""
        _config.DEFAULT_DTYPE = as_dtype(value)
        _config.DEFAULT_COMPLEX_DTYPE = as_dtype(_MAP_FLOAT_TO_COMPLEX[value])

        _update_default_dtypes()

        return get_default_dtype()

    return set_default_dtype


def _pre_cast_out_from_dtype(cast, is_floating, is_complex):
    def _cast_out_from_dtype(dtype_pos=None, target=None):
        """Cast output based on default dtype.

        Useful to wrap functions which output dtype cannot be controlled.
        """

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
                            _config.DEFAULT_DTYPE
                            if is_floating(out)
                            else _config.DEFAULT_COMPLEX_DTYPE,
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


def _pre_add_default_dtype_by_casting(cast):
    def _add_default_dtype_by_casting(target=None):
        """Add default dtype as function argument.

        Behavior is achieved by casting output (not ideal).
        """

        def _decorator(func):
            @functools.wraps(func)
            def _wrapped(*args, dtype=None, **kwargs):
                if dtype is None:
                    dtype = _config.DEFAULT_DTYPE

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


def _pre_cast_fout_to_input_dtype(cast, is_floating):
    def _cast_fout_to_input_dtype(target=None):
        """Cast out func if float and not accordingly to input.

        It is required e.g. for scipy when result is innacurate.
        """

        def _decorator(func):
            @functools.wraps(func)
            def _wrapped(x, *args, **kwargs):
                out = func(x, *args, **kwargs)
                if is_floating(out) and out.dtype != x.dtype:
                    return cast(out, x.dtype)
                return out

            return _wrapped

        if target is None:
            return _decorator
        else:
            return _decorator(target)

    return _cast_fout_to_input_dtype


def _np_box_unary_scalar(target=None):
    """Update dtype if input is float for unary operations."""

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x, *args, **kwargs):

            if type(x) is float:
                return func(x, *args, dtype=_config.DEFAULT_DTYPE, **kwargs)

            return func(x, *args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)


def _np_box_binary_scalar(target=None):
    """Update dtype if input is float for binary operations."""

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x1, x2, *args, **kwargs):

            if type(x1) is float:
                return func(x1, x2, *args, dtype=_config.DEFAULT_DTYPE, **kwargs)

            return func(x1, x2, *args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)
