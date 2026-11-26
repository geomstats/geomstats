import pytest
from polpo.testing.core import TestFunction
from polpo.testing.data import _unpack_test_datum
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.test_cases.backend._utils import get_backend_fnc


def _make_data_method(data, marks):
    """Make a data method returning given data.

    Parameters
    ----------
    data : list
        Data to return.
    marks : list[pytest.Mark]
        Marks of the data method.

    Returns
    -------
    data_method : callable
        Data method.
    """

    def data_method():
        return data

    if marks:
        data_method.pytestmark = list(marks)

    return data_method


class BackendDataBasedParametrizer(GeometricDataBasedParametrizer):
    """Parametrizer splitting backend tests by function.

    Each test with a ``func_name`` argument, e.g. ``test_func_vec``, becomes
    one test per function, e.g. ``test_linalg_cholesky_vec``, where ``func``
    is replaced by the function name with dots replaced by underscores. The
    original test is kept without data, so it is ignored.
    """

    @classmethod
    def _expand_tests(cls, tests, testing_data):
        tests = super()._expand_tests(tests, testing_data)

        new_tests = {}
        for name, test in tests.items():
            if (
                not test.active
                or "func_name" not in test.arg_names
                or not name.startswith("test_func")
            ):
                new_tests[name] = test
                continue

            groups = {}
            for datum in test.data_fnc():
                func_name = _unpack_test_datum(datum)[0]["func_name"]
                groups.setdefault(func_name, []).append(datum)

            # without data, so the inherited test is ignored
            new_tests[name] = TestFunction(name, test.func)

            suffix = name.removeprefix("test_func")
            for func_name, data in groups.items():
                new_name = f"test_{func_name.replace('.', '_')}{suffix}"
                new_test = TestFunction(new_name, test.func).set_data_method(
                    _make_data_method(data, test._data_marks)
                )
                for mark in test._marks:
                    new_test.add_mark(mark)

                new_tests[new_name] = new_test

        return new_tests


class BackendTestCase(TestCase):
    def test_func(
        self, func_name, a, expected, b=None, c=None, kwargs=None, atol=gs.atol
    ):
        """Check a function of one, two or three arrays.

        The function is called with the arguments among `a`, `b` and `c`
        that are not None, followed by `kwargs`.

        Parameters
        ----------
        func_name : str
            Name of the backend function.
        a : array-like
            First input array.
        expected : array-like
            Expected output.
        b : array-like
            Second input array, if any.
        c : array-like
            Third input array, if any.
        kwargs : dict
            Keyword arguments, if any.
        atol : float
            Absolute tolerance.
        """
        gs_fnc = get_backend_fnc(func_name)
        args = [arg for arg in (a, b, c) if arg is not None]
        res = gs_fnc(*args, **(kwargs or {}))
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.shape
    def test_func_out_shape(self, func_name, args, expected):
        """Check the shape of the output of a function.

        Parameters
        ----------
        func_name : str
            Name of the backend function.
        args : tuple
            Positional arguments of the function.
        expected : tuple
            Expected output shape.
        """
        gs_fnc = get_backend_fnc(func_name)

        shape = gs.shape(gs_fnc(*args))
        self.assertEqual(shape, expected, msg=f"{shape} instead of {expected}")

    def test_func_is_reproducible(self, func_name, args):
        """Check a random function gives the same output after the same seed.

        Parameters
        ----------
        func_name : str
            Name of the backend function, e.g. ``"random.choice"``.
        args : tuple
            Positional arguments of the function.
        """
        gs_fnc = get_backend_fnc(func_name)

        gs.random.seed(0)
        out = gs_fnc(*args)
        gs.random.seed(0)
        out_ = gs_fnc(*args)
        self.assertAllClose(out_, out)

    @pytest.mark.type
    def test_func_out_type(self, func_name, args, expected):
        """Check the type of the output of a function.

        Parameters
        ----------
        func_name : str
            Name of the backend function.
        args : tuple
            Positional arguments of the function.
        expected : type
            Expected output type.
        """
        gs_fnc = get_backend_fnc(func_name)

        out = gs_fnc(*args)
        self.assertTrue(isinstance(out, expected))

    def test_func_out_equal(self, func_name, args, expected):
        """Check the output of a function is equal to the expected one.

        Parameters
        ----------
        func_name : str
            Name of the backend function.
        args : tuple
            Positional arguments of the function.
        expected : object
            Expected output.
        """
        gs_fnc = get_backend_fnc(func_name)

        out = gs_fnc(*args)
        self.assertEqual(out, expected)

    def test_func_raises_error(self, func_name, a, b=None, c=None):
        """Check a function raises an error for incompatible inputs.

        The function is called with the arguments among `a`, `b` and `c`
        that are not None.

        Parameters
        ----------
        func_name : str
            Name of the backend function.
        a : array-like
            First input array.
        b : array-like
            Second input array, if any.
        c : array-like
            Third input array, if any.
        """
        gs_fnc = get_backend_fnc(func_name)
        args = [arg for arg in (a, b, c) if arg is not None]

        with pytest.raises((ValueError, RuntimeError)):
            gs_fnc(*args)

    def test_compose_with_inverse(
        self, func_name_1, func_name_2, a, kwargs=None, atol=gs.atol
    ):
        """Check composing a function with its inverse gives the identity.

        Parameters
        ----------
        func_name_1 : str
            Name of the backend function applied first, e.g. ``"linalg.logm"``.
        func_name_2 : str
            Name of its inverse, e.g. ``"linalg.expm"``.
        a : array-like
            Input array.
        kwargs : dict
            Keyword arguments of both functions, if any.
        atol : float
            Absolute tolerance.
        """
        gs_fnc_1 = get_backend_fnc(func_name_1)
        gs_fnc_2 = get_backend_fnc(func_name_2)
        kwargs = kwargs or {}

        out = gs_fnc_2(gs_fnc_1(a, **kwargs), **kwargs)
        self.assertAllClose(out, a, atol=atol)
