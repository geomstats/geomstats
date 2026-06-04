"""Machinery to handle global control of dtypes.

Notes
-----
Functions starting with "_pre" are shared functions that just need access to
specific backend functions. e.g. `_pre_set_default_dtype` requires access to
`as_dtype`. `set_default_dtype` can then be created in each backend by doing
`set_default_dtype = _pre_set_default_dtype(as_dtype)`. The same principle
applies to ("_pre") decorators. This decreases code duplication, while being
able to avoid (dirty) circular imports.
"""

import functools
import inspect
import types

_MAP_FLOAT_TO_COMPLEX = {
    "float32": "complex64",
    "float64": "complex128",
}


def _pre_add_default_dtype_by_casting(cast, get_default_dtype):
    def _add_default_dtype_by_casting(target=None):
        """Add default dtype as function argument.

        Behavior is achieved by casting output (not ideal, but impoosible to
        avoid without acting directly in the backends themselves).

        How it works?
        -------------
        Function is called normally. If output is float or complex, then it
        checks if is of expected dtype. If not, cast is performed.
        """

        def _decorator(func):
            @functools.wraps(func)
            def _wrapped(*args, dtype=None, **kwargs):
                if dtype is None:
                    dtype = get_default_dtype()

                out = func(*args, **kwargs)
                if out.dtype != dtype:
                    return cast(out, dtype)
                return out

            return _wrapped

        if target is None:
            return _decorator

        return _decorator(target)

    return _add_default_dtype_by_casting


def _pre_cast_out_to_input_dtype(cast, is_floating, is_complex, as_dtype, dtype_as_str):
    def _cast_out_to_input_dtype(target=None):
        """Cast out func if float or complex and not accordingly to input.

        How it works?
        -------------
        Function is called normally.
        If output is float, then it checks if is of expected dtype
        (input dtype). If not, cast is performed.
        If output is complex, then if first check if input is complex, if not
        it verifies the required precision for complex dtype and casts
        accordingly (if necessary)
        """

        def _decorator(func):
            @functools.wraps(func)
            def _wrapped(x, *args, **kwargs):
                out = func(x, *args, **kwargs)

                if is_floating(out):
                    if out.dtype != x.dtype:
                        return cast(out, x.dtype)
                elif is_complex(out):
                    if is_complex(x):
                        cmp_dtype = x.dtype
                    else:
                        float_name = dtype_as_str(x.dtype)
                        cmp_dtype = as_dtype(f"complex{int((float_name[-2:])) * 2}")

                    if out.dtype != cmp_dtype:
                        return cast(out, cmp_dtype)

                return out

            return _wrapped

        if target is None:
            return _decorator

        return _decorator(target)

    return _cast_out_to_input_dtype


def _pre_allow_complex_dtype(cast, complex_dtypes, get_default_dtype):
    def _allow_complex_dtype(target=None):
        """Allow complex type by calling the function twice.

        Assumes function do not support dtype.

        How it works?
        -------------
        Function is called twice if dtype is complex.
        Output is casted if not corresponding to expected dtype.
        """

        def _decorator(func):
            @functools.wraps(func)
            def _wrapped(*args, dtype=None, **kwargs):
                if dtype is None:
                    dtype = get_default_dtype()

                out = func(*args, **kwargs)
                if dtype in complex_dtypes:
                    out = out + 1j * func(*args, **kwargs)

                if out.dtype != dtype:
                    return cast(out, dtype)

                return out

            return _wrapped

        if target is None:
            return _decorator

        return _decorator(target)

    return _allow_complex_dtype
