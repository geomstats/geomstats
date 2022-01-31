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
        self, args, n_samples=100, rtol=gs.rtol, atol=gs.atol
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
        random_data = []
        for metric_args, space in args:
            base_point = space.random_point(n_samples)
            point = space.random_point(n_samples)
            random_data.append(((metric_args), point, base_point, rtol, atol))
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
        self, args, n_samples=100, rtol=gs.rtol, atol=gs.atol
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
        random_data = []
        for metric_args, space in args:
            print("test", metric_args)
            points_a = space.random_point(n_samples)
            points_b = space.random_point(n_samples)
            random_data.append(((metric_args), points_a, points_b, rtol, atol))
        return self.generate_tests([], random_data)

    def _exp_belongs_data(self, args, n_samples=10):
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
        random_data = []
        for metric_args, space in args:
            print("test ing", metric_args)
            # TODO  (sait) : after random_tangent_vec is fixed
            int_n_samples = (int)(gs.sqrt(n_samples))
            base_points = space.random_point(int_n_samples)
            for base_point in base_points:
                tangent_vec = space.random_tangent_vec(
                    int_n_samples, base_point=base_point
                )
                random_data.append(((metric_args), space, tangent_vec, base_point))
        return self.generate_tests([], random_data)

    def _log_is_tangent_data(self, args, n_samples=100):
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
        random_data = []
        for metric_args, space in args:
            base_points = space.random_point(n_samples)
            points = space.random_point(n_samples)
            random_data.append(((metric_args), space, base_points, points))
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
        print("")
        print("test", attrs)
        print("")

        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, types.FunctionType):

                args_str = ", ".join(inspect.getfullargspec(attr_value)[0][1:])
                data_fn_str = attr_name[5:] + "_data"
                attrs[attr_name] = pytest.mark.parametrize(
                    args_str,
                    getattr(locals()["attrs"]["testing_data"], data_fn_str)(),
                )(attr_value)

        return super(Parametrizer, cls).__new__(cls, name, bases, attrs)


class MetricParametrizer(Parametrizer):
    def __new__(cls, name, bases, attrs):
        def test_log_exp_composition(self, metric_args, point, base_point, rtol, atol):
            metric = self.cls(*metric_args)
            log = metric.log(gs.array(point), base_point=gs.array(base_point))
            result = metric.exp(tangent_vec=log, base_point=gs.array(base_point))
            self.assertAllClose(result, point, rtol=rtol, atol=atol)

        def test_squared_dist_is_symmetric(
            self, metric_args, point_a, point_b, rtol, atol
        ):
            metric = self.cls(*metric_args)
            sd_a_b = metric.squared_dist(gs.array(point_a), gs.array(point_b))
            sd_b_a = metric.squared_dist(gs.array(point_b), gs.array(point_a))
            self.assertAllClose(sd_a_b, sd_b_a, rtol=rtol, atol=atol)

        def test_exp_belongs(self, metric_args, space, tangent_vec, base_point):
            metric = self.cls(*metric_args)
            exp = metric.exp(gs.array(tangent_vec), gs.array(base_point))
            self.assertAllClose(gs.all(space.belongs(exp)), True)

        def test_log_is_tangent(self, metric_args, space, base_point, point):
            metric = self.cls(*metric_args)
            log = metric.log(gs.array(base_point), gs.array(point))
            self.assertAllClose(
                gs.all(space.is_tangent(log, gs.array(base_point))), True
            )

        attrs[test_log_exp_composition.__name__] = test_log_exp_composition
        attrs[test_squared_dist_is_symmetric.__name__] = test_squared_dist_is_symmetric
        attrs[test_exp_belongs.__name__] = test_exp_belongs
        attrs[test_log_is_tangent.__name__] = test_log_is_tangent

        return super(MetricParametrizer, cls).__new__(cls, name, bases, attrs)


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
