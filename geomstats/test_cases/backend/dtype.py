import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.test_cases.backend._utils import get_backend_fnc


class DtypeTestCase(TestCase):
    """Test case for the dtypes of backend outputs.

    Tests may change the global default dtype; it is restored to float64
    after each test, even if the test fails.
    """

    @pytest.fixture(autouse=True)
    def _restore_default_dtype(self):
        yield
        gs.set_default_dtype("float64")

    def assertDtype(self, actual, expected):
        msg = f"{actual} instead of {expected}"
        self.assertTrue(actual == expected, msg)

    def test_func_out_dtype(self, func_name, args, kwargs, expected):
        """Check the dtype of the output of a function.

        Parameters
        ----------
        func_name : str
            Name of the backend function.
        args : tuple
            Positional arguments of the function.
        kwargs : dict
            Keyword arguments of the function.
        expected : dtype or str
            Expected dtype of the output, or of each output if the function
            returns a tuple.
        """
        expected = gs.as_dtype(expected) if isinstance(expected, str) else expected

        out = get_backend_fnc(func_name)(*args, **kwargs)

        outs = out if isinstance(out, tuple) else (out,)
        for out_ in outs:
            self.assertDtype(out_.dtype, expected)

    def test_func_out_dtype_with_default_dtype(
        self, func_name, default_dtype, args=(), kwargs=None, expected=None
    ):
        """Check the output dtype of a function follows the default dtype.

        Parameters
        ----------
        func_name : str
            Name of the backend function.
        default_dtype : str
            Default dtype, set before creating the arguments.
        args : tuple or callable
            Positional arguments of the function. If callable, it creates them
            once the default dtype is set.
        kwargs : dict
            Keyword arguments of the function.
        expected : str
            Expected dtype. By default, ``default_dtype``.
        """
        gs.set_default_dtype(default_dtype)
        if callable(args):
            args = args()

        self.test_func_out_dtype(
            func_name,
            args,
            kwargs or {},
            expected=default_dtype if expected is None else expected,
        )

    def test_func_out_dtype_with_dtype(self, func_name, dtype, args=(), kwargs=None):
        """Check the output dtype of a function is the dtype it is given.

        The given dtype takes precedence over the default dtype and over the
        dtype of the inputs.

        Parameters
        ----------
        func_name : str
            Name of the backend function.
        dtype : str
            Dtype given to the function.
        args : tuple
            Positional arguments of the function.
        kwargs : dict
            Keyword arguments of the function.
        """
        self.test_func_out_dtype(
            func_name,
            args,
            {**(kwargs or {}), "dtype": gs.as_dtype(dtype)},
            expected=dtype,
        )

    def test_func_out_dtype_with_input_dtype(
        self,
        func_name,
        shapes,
        dtype,
        kwargs=None,
        create_inputs=None,
    ):
        """Check that a function keeps the dtype of its inputs.

        Parameters
        ----------
        func_name : str
            Name of the backend function.
        shapes : tuple
            Shape of each input array.
        dtype : str
            Dtype of the input arrays.
        kwargs : dict
            Keyword arguments of the function.
        create_inputs : tuple[callable]
            Function creating each input array from a shape and a dtype.
            By default, ``gs.ones``.
        """
        create_inputs = create_inputs or (gs.ones,) * len(shapes)
        args = tuple(
            create(shape, dtype=gs.as_dtype(dtype))
            for create, shape in zip(create_inputs, shapes)
        )

        self.test_func_out_dtype(func_name, args, kwargs or {}, expected=dtype)

    def test_numpy_conversion(self, shape, dtype):
        """Check conversion to and from numpy keeps the dtype.

        Parameters
        ----------
        shape : tuple
            Shape of the array.
        dtype : str
            Dtype of the array.
        """
        array = gs.ones(shape, dtype=gs.as_dtype(dtype))

        np_array = gs.to_numpy(array)
        self.assertDtype(np_array.dtype, np.dtype(dtype))

        self.assertDtype(gs.from_numpy(np_array).dtype, gs.as_dtype(dtype))

    def test_random_distrib_complex(self, func_name, args=(), kwargs=None):
        """Check a random sampler produces genuinely complex samples.

        Parameters
        ----------
        func_name : str
            Name of the backend sampler, e.g. ``"random.normal"``.
        args : tuple
            Positional arguments of the sampler.
        kwargs : dict
            Keyword arguments of the sampler.
        """
        kwargs = kwargs or {}

        gs_fnc = get_backend_fnc(func_name)

        dtype = gs.as_dtype("complex128")
        out = gs_fnc(*args, **kwargs, dtype=dtype)

        self.assertDtype(out.dtype, dtype)
        self.assertTrue(
            gs.sum(gs.abs(gs.imag(out))) > gs.atol, msg="Zero imaginary part"
        )
