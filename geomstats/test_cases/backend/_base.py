import random

import numpy as np
import pytest
import scipy

import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data

# TODO: use new strategy for vectorization tests


def _convert_gs_to_np(value):
    if gs.is_array(value):
        return gs.to_numpy(value)

    if isinstance(value, (list, tuple)):
        new_value = []
        for value_ in value:
            new_value.append(_convert_gs_to_np(value_))

        if isinstance(value, tuple):
            new_value = tuple(new_value)

        return new_value

    if isinstance(value, dict):
        return {key: _convert_gs_to_np(value_) for key, value_ in value.items()}

    return value


def convert_gs_to_np(*args):
    out = []
    for arg in args:
        val = _convert_gs_to_np(arg)
        out.append(val)

    if len(args) == 1:
        return out[0]

    return out


def get_backend_fncs(func_name, cmp_package=np):
    func_name_ls = func_name.split(".")
    gs_func = gs
    cmp_func = cmp_package
    for name in func_name_ls:
        gs_func = getattr(gs_func, name)
        cmp_func = getattr(cmp_func, name)

    return gs_func, cmp_func


def get_backend_fnc(func_name):
    func_name_ls = func_name.split(".")
    gs_func = gs
    for name in func_name_ls:
        gs_func = getattr(gs_func, name)

    return gs_func


def _get_n_reps(n_reps):
    if n_reps is None:
        return random.randint(2, 5)

    return n_reps


class _AgainstNumpyBasedTestCase(TestCase):
    def _test_func_np_based(self, package, func_name, args):
        gs_fnc, np_fnc = get_backend_fncs(func_name, cmp_package=package)
        np_args = convert_gs_to_np(args)

        gs_array = gs_fnc(*args)
        np_array = np_fnc(*np_args)

        self.assertAllCloseToNp(gs_array, np_array)


class AgainstNumpyTestCase(_AgainstNumpyBasedTestCase):
    def test_array_like_np(self, func_name, args):
        return self.test_func_like_np(func_name, args)

    def test_func_like_np(self, func_name, args):
        return self._test_func_np_based(np, func_name, args)

    def test_unary_op_like_np(self, func_name, a):
        return self.test_func_like_np(func_name, [a])

    def test_binary_op_like_np(self, func_name, a, b):
        return self.test_func_like_np(func_name, [a, b])


class AgainstScipyTestCase(_AgainstNumpyBasedTestCase):
    def _test_func_like_scipy(self, func_name, args):
        return self._test_func_np_based(scipy, func_name, args)

    def test_unary_op_like_scipy(self, func_name, a):
        return self._test_func_like_scipy(func_name, [a])


class AgainstEinsumTestCase(TestCase):
    def test_binary_op_like_einsum(self, func_name, a, b, einsum_expr):
        gs_fnc = get_backend_fnc(func_name)

        gs_out = gs_fnc(a, b)
        ein_out = gs.einsum(einsum_expr, a, b)

        self.assertAllClose(gs_out, ein_out)


class BackendTestCase(TestCase):
    def test_compose_with_inverse(self, func_name_1, func_name_2, a):
        gs_fnc_1 = get_backend_fnc(func_name_1)
        gs_fnc_2 = get_backend_fnc(func_name_2)

        out = gs_fnc_2(gs_fnc_1(a))
        self.assertAllClose(out, a)

    def test_unary_op(self, func_name, a, expected, atol):
        gs_fnc = get_backend_fnc(func_name)
        res = gs_fnc(a)
        self.assertAllClose(res, expected, atol)

    @pytest.mark.vec
    def test_unary_op_vec(self, func_name, a, atol, n_reps=None):
        n_reps = _get_n_reps(n_reps)
        gs_fnc = get_backend_fnc(func_name)
        expected = gs_fnc(a)

        vec_data = generate_vectorization_data(
            data=[dict(func_name=func_name, a=a, expected=expected, atol=atol)],
            arg_names=["a"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_binary_op(self, func_name, a, b, expected, atol):
        gs_fnc = get_backend_fnc(func_name)
        res = gs_fnc(a, b)
        self.assertAllClose(res, expected, atol)

    @pytest.mark.vec
    def test_binary_op_vec(self, func_name, a, b, atol, n_reps=None):
        n_reps = _get_n_reps(n_reps)
        gs_fnc = get_backend_fnc(func_name)
        expected = gs_fnc(a, b)

        vec_data = generate_vectorization_data(
            data=[dict(func_name=func_name, a=a, b=b, expected=expected, atol=atol)],
            arg_names=["a", "b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_binary_op_raises_error(self, func_name, a, b):
        gs_fnc = get_backend_fnc(func_name)

        with pytest.raises(Exception):
            gs_fnc(a, b)

    @pytest.mark.shape
    def test_func_out_shape(self, func_name, args, expected):
        gs_fnc = get_backend_fnc(func_name)

        # TODO: better comparison (for more info when failing)
        out = gs_fnc(*args)
        self.assertTrue(gs.shape(out) == expected)

    @pytest.mark.type
    def test_func_out_type(self, func_name, args, expected):
        gs_fnc = get_backend_fnc(func_name)

        out = gs_fnc(*args)
        self.assertTrue(isinstance(out, expected))

    def test_func_out_bool(self, func_name, args, expected):
        gs_fnc = get_backend_fnc(func_name)

        out = gs_fnc(*args)
        if expected:
            self.assertTrue(out)
        else:
            self.assertFalse(out)

    def test_func_out_allclose(self, func_name, expected, args=(), kwargs=None):
        kwargs = kwargs or {}
        gs_fnc = get_backend_fnc(func_name)

        out = gs_fnc(*args, **kwargs)
        self.assertAllClose(out, expected)

    def test_func_out_equal(self, func_name, args, expected):
        gs_fnc = get_backend_fnc(func_name)

        out = gs_fnc(*args)
        self.assertEqual(out, expected)


@pytest.mark.type
class DtypeTestCase(TestCase):
    dtypes_str = ["float32", "float64"]  # sort by wider
    default_dtype = gs.as_dtype("float64")

    def _reset_dtype(self):
        gs.set_default_dtype("float64")

    def assertDtype(self, actual, expected):
        self._reset_dtype()

        msg = f"{actual} instead of {expected}"
        self.assertTrue(actual == expected, msg)

    def test_array(self, ls, global_dtype_str, expected_dtype):
        gs.set_default_dtype(global_dtype_str)

        out = gs.array(ls)
        self.assertDtype(out.dtype, expected_dtype)

    def test_array_creation(self, func_name, args, kwargs):
        gs_fnc = get_backend_fnc(func_name)

        for dtype_str in self.dtypes_str:
            dtype = gs.set_default_dtype(dtype_str)
            out = gs_fnc(*args, **kwargs)
            self.assertDtype(out.dtype, dtype)

    def test_array_creation_with_dtype(self, func_name, args=(), kwargs=None):
        kwargs = kwargs or {}
        gs_fnc = get_backend_fnc(func_name)

        for dtype_str in self.dtypes_str:
            # test global
            dtype = gs.set_default_dtype(dtype_str)
            out = gs_fnc(*args, **kwargs)
            self.assertDtype(out.dtype, dtype)

            # test specifying dtype
            for dtype_inner_str in self.dtypes_str:
                dtype_inner = gs.as_dtype(dtype_inner_str)
                out = gs_fnc(*args, dtype=dtype_inner, **kwargs)
                self.assertDtype(out.dtype, dtype_inner)

    def test_array_creation_with_dtype_from_shape(self, func_name, shape):
        return self.test_array_creation_with_dtype(func_name, (shape,), {})

    def test_unary_op_float_input(self, func_name, x):
        gs_fnc = get_backend_fnc(func_name)

        for dtype_str in self.dtypes_str:
            dtype = gs.set_default_dtype(dtype_str)
            out = gs_fnc(x)

            self.assertDtype(out.dtype, dtype)

    def test_array_creation_with_dtype_from_array(
        self,
        func_name,
        array_shape,
        kwargs=None,
        func_array=gs.ones,
    ):
        self._reset_dtype()

        kwargs = kwargs or {}
        gs_fnc = get_backend_fnc(func_name)

        for dtype_str in self.dtypes_str:
            # test dynamic
            dtype = gs.as_dtype(dtype_str)
            a = func_array(array_shape, dtype=dtype)

            out = gs_fnc(a, **kwargs)
            self.assertDtype(out.dtype, a.dtype)

            # test specifying dtype
            for dtype_inner_str in self.dtypes_str:
                dtype_inner = gs.as_dtype(dtype_inner_str)
                out = gs_fnc(a, dtype=dtype_inner, **kwargs)
                self.assertDtype(out.dtype, dtype_inner)

    def test_from_numpy_given_shape(
        self, array_shape, kwargs=None, np_func_array=gs.ones
    ):
        self._reset_dtype()

        func_name = "from_numpy"
        kwargs = kwargs or {}
        gs_fnc = get_backend_fnc(func_name)

        for dtype_str in self.dtypes_str:
            # test dynamic
            np_dtype = np.dtype(dtype_str)
            dtype = gs.as_dtype(dtype_str)
            a = np_func_array(array_shape, dtype=np_dtype)

            out = gs_fnc(a, **kwargs)
            self.assertDtype(out.dtype, dtype)

    def test_to_numpy_given_shape(self, array_shape, kwargs=None, func_array=gs.ones):
        self._reset_dtype()

        func_name = "to_numpy"

        kwargs = kwargs or {}
        gs_fnc = get_backend_fnc(func_name)

        for dtype_str in self.dtypes_str:
            np_dtype = np.dtype(dtype_str)
            dtype = gs.as_dtype(dtype_str)
            a = func_array(array_shape, dtype=dtype)

            out = gs_fnc(a, **kwargs)
            self.assertDtype(out.dtype, np_dtype)

    def test_unary_op_with_dtype_given_shape(self, func_name, array_shape, kwargs=None):
        return self.test_array_creation_with_dtype_from_array(
            func_name, array_shape, kwargs=kwargs
        )

    def test_unary_op_given_shape(
        self, func_name, array_shape, kwargs=None, func_array=gs.ones
    ):
        self._reset_dtype()

        kwargs = kwargs or {}
        gs_fnc = get_backend_fnc(func_name)

        for dtype_str in self.dtypes_str:
            dtype = gs.as_dtype(dtype_str)
            a = func_array(array_shape, dtype=dtype)

            out = gs_fnc(a, **kwargs)
            self.assertDtype(out.dtype, dtype)

    def test_unary_op_mult_out_given_shape(
        self, func_name, array_shape, func_array=gs.ones
    ):
        self._reset_dtype()

        gs_fnc = get_backend_fnc(func_name)

        for dtype_str in self.dtypes_str:
            dtype = gs.as_dtype(dtype_str)
            a = func_array(array_shape, dtype=dtype)

            out = gs_fnc(a)
            for out_ in out:
                self.assertDtype(out_.dtype, dtype)

    def _test_op_given_array(self, func_name, create_array):
        # create_array to avoid using cast
        gs_fnc = get_backend_fnc(func_name)

        for dtype_str in self.dtypes_str:
            dtype = gs.set_default_dtype(dtype_str)
            args = create_array()
            if gs.is_array(args):
                args = [args]

            out = gs_fnc(*args)
            self.assertDtype(out.dtype, dtype)

    def test_unary_op_given_array(self, func_name, create_array):
        return self._test_op_given_array(func_name, create_array)

    def test_binary_op_float_input(self, func_name, x1, x2):
        gs_fnc = get_backend_fnc(func_name)

        for dtype_str in self.dtypes_str:
            dtype = gs.set_default_dtype(dtype_str)
            out = gs_fnc(x1, x2)

            self.assertDtype(out.dtype, dtype)

    def test_binary_op_given_shape(
        self, func_name, shape_a, shape_b, kwargs=None, func_a=gs.ones, func_b=gs.ones
    ):
        self._reset_dtype()

        kwargs = kwargs or {}
        gs_fnc = get_backend_fnc(func_name)

        for i, dtype_a_str in enumerate(self.dtypes_str):
            dtype_a = gs.as_dtype(dtype_a_str)

            a = func_a(shape_a, dtype=dtype_a)

            for j, dtype_b_str in enumerate(self.dtypes_str):
                dtype_b = gs.as_dtype(dtype_b_str)

                b = func_b(shape_b, dtype=dtype_b)

                out = gs_fnc(a, b, **kwargs)
                cmp_dtype = dtype_a if i > j else dtype_b

                self.assertDtype(out.dtype, cmp_dtype)

    def test_ternary_op_given_shape(
        self,
        func_name,
        shape_a,
        shape_b,
        shape_c,
        func_a=gs.ones,
        func_b=gs.ones,
        func_c=gs.ones,
    ):
        self._reset_dtype()

        gs_fnc = get_backend_fnc(func_name)

        for i, dtype_a_str in enumerate(self.dtypes_str):
            dtype_a = gs.as_dtype(dtype_a_str)

            a = func_a(shape_a, dtype=dtype_a)

            for j, dtype_b_str in enumerate(self.dtypes_str):
                dtype_b = gs.as_dtype(dtype_b_str)

                b = func_b(shape_b, dtype=dtype_b)

                for k, dtype_c_str in enumerate(self.dtypes_str):
                    dtype_c = gs.as_dtype(dtype_c_str)
                    c = func_c(shape_c, dtype=dtype_c)

                    out = gs_fnc(a, b, c)
                    cmp_dtype = gs.as_dtype(self.dtypes_str[max([i, j, k])])

                    self.assertDtype(out.dtype, cmp_dtype)

    def test_ternary_op_given_array(self, func_name, create_array):
        return self._test_op_given_array(func_name, create_array)

    def test_func_out_dtype(self, func_name, args, kwargs, expected):
        self._reset_dtype()

        gs_fnc = get_backend_fnc(func_name)

        out = gs_fnc(*args, **kwargs)

        self.assertDtype(out.dtype, expected)

    def test_solve_sylvester(
        self,
        shape_a,
        shape_b,
        shape_c,
        func_a=gs.ones,
        func_b=gs.ones,
        func_c=gs.ones,
    ):
        func_name = "linalg.solve_sylvester"

        self.test_ternary_op_given_shape(
            func_name,
            shape_a,
            shape_b,
            shape_c,
            func_a=func_a,
            func_b=func_b,
            func_c=func_c,
        )

    def test_random_distrib_complex(self, func_name, args=(), kwargs=None):
        self._reset_dtype()
        kwargs = kwargs or {}

        gs_fnc = get_backend_fnc(func_name)

        dtype = gs.as_dtype("complex128")
        out = gs_fnc(*args, **kwargs, dtype=dtype)

        self.assertDtype(out.dtype, dtype)
        self.assertTrue(
            gs.sum(gs.abs(gs.imag(out))) > gs.atol, msg="Zero imaginary part"
        )
