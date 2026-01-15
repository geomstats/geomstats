"""Unit tests for automatic differentiation in different backends."""

import warnings

import pytest

import geomstats.backend as gs
from geomstats.geometry.grassmannian import Grassmannian
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import TestCase, autograd_and_torch_only, np_only
from geomstats.test_cases.backend.autodiff import (
    AutodiffTestCase,
    MetricDistGradTestCase,
    NumpyRaisesTestCase,
)

from .data.autodiff import (
    CustomGradientTestData,
    MetricDistGradTestData,
    NewAutodiffTestData,
    NumpyRaisesTestData,
)


def _sphere_immersion(point):
    """Sphere immersion

    Parameters
    ----------
    point : array-like, shape=[2,]

    Returns
    -------
    immersed_point : array-like, shape=[3,]
    """
    radius = 4.0
    theta, phi = point
    x = gs.sin(theta) * gs.cos(phi)
    y = gs.sin(theta) * gs.sin(phi)
    z = gs.cos(theta)
    return gs.stack([radius * x, radius * y, radius * z])


def _first_component_of_sphere_immersion(point):
    """First component of the sphere immersion function.

    This returns a vector of dim 1.

    Parameters
    ----------
    point : array-like, shape=[2,]
    """
    radius = 4.0
    theta, phi = point
    x = gs.sin(theta) * gs.cos(phi)
    return gs.array([radius * x])


def _first_component_of_sphere_immersion_scalar(point):
    """First component of the sphere immersion function.

    This returns a scalar.
    """
    radius = 4.0
    theta, phi = point
    x = gs.sin(theta) * gs.cos(phi)
    return radius * x


@np_only
class TestNumpyRaises(NumpyRaisesTestCase, metaclass=DataBasedParametrizer):
    dummy_func = lambda v: gs.sum(v**2)
    testing_data = NumpyRaisesTestData()


@autograd_and_torch_only
class TestAutodiff(AutodiffTestCase, metaclass=DataBasedParametrizer):
    testing_data = NewAutodiffTestData()


@pytest.fixture(
    scope="class",
    params=[
        SpecialEuclidean(3),
        Grassmannian(3, 2),
    ],
)
def equipped_spaces(request):
    request.cls.space = request.param


@autograd_and_torch_only
class TestCustomGradient(AutodiffTestCase, metaclass=DataBasedParametrizer):
    testing_data = CustomGradientTestData()


@autograd_and_torch_only
@pytest.mark.usefixtures("equipped_spaces")
class TestMetricDistGrad(MetricDistGradTestCase, metaclass=DataBasedParametrizer):
    testing_data = MetricDistGradTestData()


class TestAutodiffOld(TestCase):
    def setup_method(self):
        warnings.simplefilter("ignore", category=ImportWarning)
        self.n_samples = 2

    @autograd_and_torch_only
    def test_jacobian(self):
        """Test that jacobians are consistent across backends.

        The jacobian of a function f going from an input space A to an output
        space B is a matrix of shape (dim_B, dim_A).
        - The columns index the derivatives wrt. the coordinates of the input space A.
        - The rows index the coordinates of the output space B.
        """
        radius = 4.0
        embedding_dim, dim = 3, 2

        point = gs.array([gs.pi / 3, gs.pi])
        theta = point[0]
        phi = point[1]
        jacobian_ai = gs.autodiff.jacobian(_sphere_immersion)(point)

        expected_1i = gs.array(
            [
                radius * gs.cos(theta) * gs.cos(phi),
                -radius * gs.sin(theta) * gs.sin(phi),
            ]
        )
        expected_2i = gs.array(
            [
                radius * gs.cos(theta) * gs.sin(phi),
                radius * gs.sin(theta) * gs.cos(phi),
            ]
        )
        expected_3i = gs.array(
            [
                -radius * gs.sin(theta),
                0,
            ]
        )
        expected_ai = gs.stack([expected_1i, expected_2i, expected_3i], axis=0)
        self.assertAllClose(jacobian_ai.shape, (embedding_dim, dim))
        self.assertAllClose(jacobian_ai.shape, expected_ai.shape)
        self.assertAllClose(jacobian_ai, expected_ai)

    @autograd_and_torch_only
    def test_jacobian_vec(self):
        """Test that jacobian_vec is correctly vectorized.

        The autodiff jacobian is not vectorized by default in torch, tf and autograd.

        The jacobian of a function f going from an input space A to an output
        space B is a matrix of shape (dim_B, dim_A).

        - The columns index the derivatives wrt. the coordinates of the input space A.
        - The rows index the coordinates of the output space B.
        """
        radius = 4.0
        embedding_dim, dim = 3, 2

        points = gs.array([[gs.pi / 3, gs.pi], [gs.pi / 5, gs.pi / 2]])
        thetas = points[:, 0]
        phis = points[:, 1]
        jacobian_ai = gs.autodiff.jacobian_vec(_sphere_immersion)(points)

        expected_1i = gs.stack(
            [
                gs.array(
                    [
                        radius * gs.cos(theta) * gs.cos(phi),
                        -radius * gs.sin(theta) * gs.sin(phi),
                    ]
                )
                for theta, phi in zip(thetas, phis)
            ]
        )
        expected_2i = gs.stack(
            [
                gs.array(
                    [
                        radius * gs.cos(theta) * gs.sin(phi),
                        radius * gs.sin(theta) * gs.cos(phi),
                    ]
                )
                for theta, phi in zip(thetas, phis)
            ]
        )
        expected_3i = gs.stack(
            [
                gs.array(
                    [
                        -radius * gs.sin(theta),
                        0,
                    ]
                )
                for theta in thetas
            ]
        )
        expected_ai = gs.stack([expected_1i, expected_2i, expected_3i], axis=1)
        self.assertAllClose(jacobian_ai.shape, (len(points), embedding_dim, dim))
        self.assertAllClose(jacobian_ai.shape, expected_ai.shape)
        self.assertAllClose(jacobian_ai, expected_ai)

    @autograd_and_torch_only
    def test_hessian(self):
        radius = 4.0
        dim = 2

        point = gs.array([gs.pi / 3, gs.pi])
        theta = point[0]
        phi = point[1]
        hessian_1ij = gs.autodiff.hessian(_first_component_of_sphere_immersion_scalar)(
            point
        )

        expected_1ij = radius * gs.array(
            [
                [-gs.sin(theta) * gs.cos(phi), -gs.cos(theta) * gs.sin(phi)],
                [-gs.cos(theta) * gs.sin(phi), -gs.sin(theta) * gs.cos(phi)],
            ]
        )

        self.assertAllClose(hessian_1ij.shape, (dim, dim))
        self.assertAllClose(hessian_1ij.shape, expected_1ij.shape)
        self.assertAllClose(hessian_1ij, expected_1ij)

    @autograd_and_torch_only
    def test_hessian_vec(self):
        """Hessian is not vectorized by default in torch, tf and autograd."""
        radius = 4.0
        dim = 2

        points = gs.array([[gs.pi / 3, gs.pi], [gs.pi / 4, gs.pi / 2]])
        thetas = points[:, 0]
        phis = points[:, 1]
        hessian_1ij = gs.autodiff.hessian_vec(
            _first_component_of_sphere_immersion_scalar
        )(points)

        expected_1ij = gs.stack(
            [
                radius
                * gs.array(
                    [
                        [-gs.sin(theta) * gs.cos(phi), -gs.cos(theta) * gs.sin(phi)],
                        [-gs.cos(theta) * gs.sin(phi), -gs.sin(theta) * gs.cos(phi)],
                    ]
                )
                for theta, phi in zip(thetas, phis)
            ],
            axis=0,
        )

        self.assertAllClose(hessian_1ij.shape, (2, dim, dim))
        self.assertAllClose(hessian_1ij.shape, expected_1ij.shape)
        self.assertAllClose(hessian_1ij, expected_1ij)
