"""Pytest utility classes, functions and fixtures."""

import inspect
import itertools
import os
import random
import types

import numpy as np
import pytest

import geomstats.backend as gs


def autograd_backend():
    """Check if autograd is set as backend."""
    return os.environ["GEOMSTATS_BACKEND"] == "autograd"


def np_backend():
    """Check if numpy is set as backend."""
    return os.environ["GEOMSTATS_BACKEND"] == "numpy"


def pytorch_backend():
    """Check if pytorch is set as backend."""
    return os.environ["GEOMSTATS_BACKEND"] == "pytorch"


def tf_backend():
    """Check if tensorflow is set as backend."""
    return os.environ["GEOMSTATS_BACKEND"] == "tensorflow"


if tf_backend():
    import tensorflow as tf

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if pytorch_backend():
    import torch


autograd_only = pytest.mark.skipif(
    not autograd_backend(), reason="Test for autograd backend only."
)
np_only = pytest.mark.skipif(not np_backend(), reason="Test for numpy backend only.")
torch_only = pytest.mark.skipif(
    not pytorch_backend(), reason="Test for pytorch backends only."
)
tf_only = pytest.mark.skipif(
    not tf_backend(), reason="Test for tensorflow backends only."
)

np_and_tf_only = pytest.mark.skipif(
    not (np_backend() or tf_backend()),
    reason="Test for numpy and tensorflow backends only.",
)
np_and_torch_only = pytest.mark.skipif(
    not (np_backend() or pytorch_backend()),
    reason="Test for numpy and pytorch backends only.",
)
np_and_autograd_only = pytest.mark.skipif(
    not (np_backend() or autograd_backend()),
    reason="Test for numpy and autograd backends only.",
)
autograd_and_torch_only = pytest.mark.skipif(
    not (autograd_backend() or pytorch_backend()),
    reason="Test for autograd and torch backends only.",
)
autograd_and_tf_only = pytest.mark.skipif(
    not (autograd_backend() or tf_backend()),
    reason="Test for autograd and tf backends only.",
)

np_autograd_and_tf_only = pytest.mark.skipif(
    not (np_backend() or autograd_backend() or tf_backend()),
    reason="Test for numpy, autograd and tensorflow backends only.",
)
np_autograd_and_torch_only = pytest.mark.skipif(
    not (np_backend() or autograd_backend() or pytorch_backend()),
    reason="Test for numpy, autograd and pytorch backends only.",
)
autograd_tf_and_torch_only = pytest.mark.skipif(
    np_backend(), reason="Test for backends with automatic differentiation only."
)


def pytorch_error_msg(a, b, rtol, atol):
    msg = f"\ntensor 1\n{a}\ntensor 2\n{b}"
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.dtype == torch.bool and b.dtype == torch.bool:
            diff = torch.logical_xor(a, b)
            msg = msg + f"\ndifference \n{diff}"
        else:
            diff = torch.abs(a - b)
            msg = msg + f"\ndifference \n{diff}\nrtol {rtol}\natol {atol}"
    return msg


class TestData:
    """Class for TestData objects."""

    def generate_tests(self, smoke_test_data, random_test_data=[]):
        """Wrap test data with corresponding markers.

        Parameters
        ----------
        smoke_test_data : list
            Test data that will be marked as smoke.

        random_test_data : list
            Test data that will be marked as random.
            Optional, default: []

        Returns
        -------
        _: list
            Tests.
        """
        tests = []
        if smoke_test_data:
            smoke_tests = [
                pytest.param(*data.values(), marks=pytest.mark.smoke)
                for data in smoke_test_data
            ]
            tests += smoke_tests
        if random_test_data:
            random_tests = [
                pytest.param(*data.values(), marks=pytest.mark.random)
                if isinstance(data, dict)
                else pytest.param(*data, marks=pytest.mark.random)
                for data in random_test_data
            ]
            tests += random_tests
        return tests

    def _log_exp_composition_data(
        self, space, n_samples=100, max_n=10, n_n=5, **kwargs
    ):
        """Generate Data that checks for log and exp are inverse. Specifically

            :math: `Exp_{base_point}(Log_{base_point}(point)) = point`


        Parameters
        ----------
        space : cls
            Manifold class on which metric is present.
        max_n : int
            Maximum value when generating 'n'.
            Optional, default: 20
        n_n : int
            Number of 'n' to be generated.
            Optional, default: 5
        n_samples : int
            Optional, default: 100

        Returns
        -------
        _ : list
            Test Data.
        """
        random_n = random.sample(range(1, max_n), n_n)
        random_data = []
        for n in random_n:
            for prod in itertools.product(*kwargs.values()):
                space_n = space(n)
                base_point = space_n.random_point(n_samples)
                point = space_n.random_point(n_samples)
                random_data.append((n,) + prod + (point, base_point))
        return self.generate_tests([], random_data)

    def _geodesic_belongs_data(
        self, space, max_n=10, n_n=5, n_geodesics=10, n_t=10, **kwargs
    ):
        """Generate Data that checks for points on geodesic belongs to data.

        Parameters
        ----------
        space : cls
            Manifold class on which metric is present.
        max_n : int
            Maximum value when generating 'n'.
            Optional, default: 10
        n_n : int
            Maximum value when generating 'n'.
            Optional, default: 5
        n_geodesics : int
            Number of geodesics to be generated.
            Optional, default: 10
        n_t : int
            Number of points to be sampled on each geodesic.
            Optional, default: 10
        Returns
        -------
        _ : list
            Test Data.
        """
        random_n = random.sample(range(2, max_n), n_n)
        random_data = []
        for n in random_n:
            for prod in itertools.product(*kwargs.values()):
                space_n = space(n)
                initial_point = space_n.random_point()
                initial_tangent_points = space_n.random_tangent_vec(
                    n_geodesics, base_point=initial_point
                )
                random_t = gs.linspace(start=-1.0, stop=1.0, num=n_t)
                for initial_tangent_point, t in itertools.product(
                    initial_tangent_points, random_t
                ):
                    random_data.append(
                        (n,) + prod + (initial_point, initial_tangent_point, t)
                    )
        return self.generate_tests([], random_data)

    def _squared_dist_is_symmetric_data(
        self, space, max_n=5, n_n=3, n_samples=10, **kwargs
    ):
        """Generate Data that checks squared_dist is symmetric.

        Parameters
        ----------
        space : cls
            Manifold class on which metric is present.
        max_n : int
            Range of 'n' to generated.
            Optional, default: 10
        n_n : int
            Maximum value when generating 'n'.
            Optional, default: 3
        n_samples : int
            Number of points to be generated.
            Optional, default: 10
        Returns
        -------
        _ : list
            Test Data.
        """
        random_n = random.sample(range(2, max_n), n_n)
        random_data = []
        for n in random_n:
            for prod in itertools.product(*kwargs.values()):
                space_n = space(n)
                points_a = space_n.random_point(n_samples)
                points_b = space_n.random_point(n_samples)
                for point_a, point_b in itertools.product(points_a, points_b):
                    random_data.append((n,) + prod + (point_a, point_b))
        return self.generate_tests([], random_data)


class Parametrizer(type):
    """Metaclass for test files.

    Parametrizer decorates every function inside the class with pytest.mark.parametrizer
    (except class methods and static methods). Two conventions need to be respected:

    1.There should be a TestData object named 'testing_data'.
    2.Every test function should have its corresponding data function inside
    TestData object.

    Ex. test_exp() should have method called exp_data() inside 'testing_data'.
    """

    def __new__(cls, name, bases, attrs):
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, types.FunctionType):

                args_str = ", ".join(inspect.getfullargspec(attr_value)[0][1:])
                data_fn_str = attr_name[5:] + "_data"
                attrs[attr_name] = pytest.mark.parametrize(
                    args_str,
                    getattr(locals()["attrs"]["testing_data"], data_fn_str)(),
                )(attr_value)

        return super(Parametrizer, cls).__new__(cls, name, bases, attrs)


class TestCase:
    """Class for Geomstats tests."""

    def assertAllClose(self, a, b, rtol=gs.rtol, atol=gs.atol):
        if tf_backend():
            return tf.test.TestCase().assertAllClose(a, b, rtol=rtol, atol=atol)
        if np_backend() or autograd_backend():
            return np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

        return self.assertTrue(
            gs.allclose(a, b, rtol=rtol, atol=atol),
            msg=pytorch_error_msg(a, b, rtol, atol),
        )

    def assertTrue(self, condition, msg=None):
        assert condition, msg

    def assertFalse(self, condition, msg=None):
        assert not condition, msg

    def assertEqual(self, a, b):
        assert a == b

    def assertAllCloseToNp(self, a, np_a, rtol=gs.rtol, atol=gs.atol):
        are_same_shape = np.all(a.shape == np_a.shape)
        are_same = np.allclose(a, np_a, rtol=rtol, atol=atol)
        assert are_same and are_same_shape

    def assertShapeEqual(self, a, b):
        if tf_backend():
            return tf.test.TestCase().assertShapeEqual(a, b)
        assert a.shape == b.shape
