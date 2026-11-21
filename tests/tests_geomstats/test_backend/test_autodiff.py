"""Unit tests for automatic differentiation in different backends."""

import warnings

import pytest

import geomstats.backend as gs
from geomstats.geometry.grassmannian import Grassmannian
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import (
    TestCase,
    autograd_and_torch_only,
    np_only,
    pytorch_backend,
    torch_only,
)
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


def _expected_hessian_first_component_of_sphere_immersion(point):
    """Closed-form hessian of `_first_component_of_sphere_immersion_scalar`."""
    radius = 4.0
    theta, phi = point[0], point[1]
    return radius * gs.array(
        [
            [-gs.sin(theta) * gs.cos(phi), -gs.cos(theta) * gs.sin(phi)],
            [-gs.cos(theta) * gs.sin(phi), -gs.sin(theta) * gs.cos(phi)],
        ]
    )


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
        jacobian_ai = gs.autodiff.jacobian(_sphere_immersion)(points)

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
        hessian_1ij = gs.autodiff.hessian(
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

    @autograd_and_torch_only
    def test_jacobian_of_jacobian_is_hessian(self):
        """Derivatives compose: nested jacobians equal the hessian (#1801)."""
        point = gs.array([gs.pi / 3, gs.pi])

        jacobian_of_jacobian = gs.autodiff.jacobian(
            gs.autodiff.jacobian(_first_component_of_sphere_immersion_scalar)
        )(point)

        expected = _expected_hessian_first_component_of_sphere_immersion(point)
        self.assertAllClose(jacobian_of_jacobian, expected)

    @autograd_and_torch_only
    def test_jacobian_vec_of_jacobian_vec_is_hessian_vec(self):
        """Nested vectorized jacobians equal the vectorized hessian (#1801)."""
        points = gs.array([[gs.pi / 3, gs.pi], [gs.pi / 4, gs.pi / 2]])

        jacobian_of_jacobian = gs.autodiff.jacobian(
            gs.autodiff.jacobian(_first_component_of_sphere_immersion_scalar)
        )(points)

        expected = gs.stack(
            [
                _expected_hessian_first_component_of_sphere_immersion(point)
                for point in points
            ]
        )
        self.assertAllClose(jacobian_of_jacobian, expected)

    @autograd_and_torch_only
    def test_jacobian_of_hessian_with_constant_hessian(self):
        """Nested derivatives of a function with constant hessian are zero."""
        point = gs.array([0.3, -1.2])

        third_derivative = gs.autodiff.jacobian(
            gs.autodiff.hessian(lambda x: x[0] * x[1])
        )(point)

        self.assertAllClose(third_derivative, gs.zeros((2, 2, 2)))

    @autograd_and_torch_only
    def test_jacobian_vec_of_hessian_vec(self):
        """Third derivative through jacobian_vec(hessian_vec(f)) is non-trivial.

        This is the nesting depth used by riemann_tensor on a PullbackMetric
        (#1928): d(christoffels) -> d(hessian_immersion) -> d(d(immersion)).
        """
        radius = 4.0
        point = gs.array([gs.pi / 3, gs.pi])
        theta, phi = point[0], point[1]

        third_derivative = gs.autodiff.jacobian(
            gs.autodiff.hessian(_first_component_of_sphere_immersion_scalar)
        )(point)

        expected_d_theta = radius * gs.array(
            [
                [-gs.cos(theta) * gs.cos(phi), gs.sin(theta) * gs.sin(phi)],
                [gs.sin(theta) * gs.sin(phi), -gs.cos(theta) * gs.cos(phi)],
            ]
        )
        expected_d_phi = radius * gs.array(
            [
                [gs.sin(theta) * gs.sin(phi), -gs.cos(theta) * gs.cos(phi)],
                [-gs.cos(theta) * gs.cos(phi), gs.sin(theta) * gs.sin(phi)],
            ]
        )

        self.assertAllClose(third_derivative.shape, (2, 2, 2))
        self.assertAllClose(third_derivative[..., 0], expected_d_theta)
        self.assertAllClose(third_derivative[..., 1], expected_d_phi)

    @autograd_and_torch_only
    def test_hessian_of_expm(self):
        """Second derivatives go through the matrix exponential.

        Around 0, sum(expm(x)) is sum(x @ x) / 2 at second order.
        """
        hessian = gs.autodiff.hessian(
            lambda x: gs.sum(gs.linalg.expm(x), axis=(-2, -1)), point_ndim=2
        )(gs.zeros((2, 2)))

        eye = gs.eye(2)
        expected = (eye[None, :, :, None] + eye[:, None, None, :]) / 2
        self.assertAllClose(hessian, expected)

    @autograd_and_torch_only
    def test_hessian_of_logm(self):
        """Second derivatives go through the matrix logarithm.

        Around the identity, sum(logm(eye + x)) is -sum(x @ x) / 2 at second order.
        """
        hessian = gs.autodiff.hessian(
            lambda x: gs.sum(gs.linalg.logm(x), axis=(-2, -1)), point_ndim=2
        )(gs.eye(2))

        eye = gs.eye(2)
        expected = -(eye[None, :, :, None] + eye[:, None, None, :]) / 2
        self.assertAllClose(hessian, expected)

    @autograd_and_torch_only
    def test_grad_of_diagonal(self):
        """The gradient of a weighted sum of the diagonal has the weights on it."""
        weights = gs.array([1.0, 2.0])

        def weighted_diagonal_sum(x):
            return gs.sum(weights * gs.diagonal(x), axis=-1)

        points = gs.reshape(gs.arange(12.0), (2, 2, 3))
        value, grad = gs.autodiff.value_and_grad(weighted_diagonal_sum, point_ndims=2)(
            points
        )

        expected_grad = gs.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        self.assertAllClose(value, gs.array([8.0, 26.0]))
        self.assertAllClose(grad, gs.stack([expected_grad, expected_grad]))

    @torch_only
    def test_grad_through_scipy_raises(self):
        """Differentiating a function scipy computes raises, as with PyTorch."""
        point = gs.array([[2.0, 1.0], [1.0, 2.0]])

        def fractional_matrix_power_sum(x):
            return gs.sum(gs.linalg.fractional_matrix_power(x, 0.5), axis=(-2, -1))

        with pytest.raises(NotImplementedError):
            gs.autodiff.value_and_grad(fractional_matrix_power_sum, point_ndims=2)(
                point
            )

    @autograd_and_torch_only
    def test_grad_of_sqrtm(self):
        """Gradients go through the matrix square root.

        With S the square root of x, the gradient of sum(S) solves
        S^T Y + Y S^T = 1. The square root of diag(-4, -9) is diag(2i, 3i),
        and the gradient of the real part is imaginary, with a sign depending
        on the complex convention.
        """

        def sqrtm_sum(x):
            return gs.sum(gs.linalg.sqrtm(x), axis=(-2, -1))

        def real_sqrtm_sum(x):
            return gs.real(sqrtm_sum(x))

        expected_grad = gs.array([[1.0 / 4.0, 1.0 / 5.0], [1.0 / 5.0, 1.0 / 6.0]])

        value, grad = gs.autodiff.value_and_grad(sqrtm_sum, point_ndims=2)(
            gs.array([[4.0, 0.0], [0.0, 9.0]])
        )
        self.assertAllClose(value, gs.array(5.0))
        self.assertAllClose(grad, expected_grad)

        value, grad = gs.autodiff.value_and_grad(real_sqrtm_sum, point_ndims=2)(
            gs.array([[-4.0, 0.0], [0.0, -9.0]], dtype=gs.get_default_cdtype())
        )
        self.assertAllClose(value, gs.array(0.0))
        self.assertAllClose(grad, (1j if pytorch_backend() else -1j) * expected_grad)

    @autograd_and_torch_only
    def test_grad_of_clip(self):
        """Clipped entries get no gradient, with keyword bounds."""
        value, grad = gs.autodiff.value_and_grad(
            lambda x: gs.sum(gs.clip(x, min=0.5), axis=-1)
        )(gs.array([[0.0, 1.0], [2.0, 0.2]]))

        self.assertAllClose(value, gs.array([1.5, 2.5]))
        self.assertAllClose(grad, gs.array([[0.0, 1.0], [1.0, 0.0]]))

    @autograd_and_torch_only
    def test_grad_of_scatter_add(self):
        """The gradient of a weighted sum of scattered values gathers the weights."""
        weights = gs.array([1.0, 2.0, 3.0])

        def weighted_scatter_add_sum(src):
            index = gs.broadcast_to(gs.array([0, 2, 0]), src.shape)
            out = gs.scatter_add(gs.zeros(src.shape), dim=-1, index=index, src=src)
            return gs.sum(weights * out, axis=-1)

        src = gs.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
        value, grad = gs.autodiff.value_and_grad(weighted_scatter_add_sum)(src)

        self.assertAllClose(value, gs.array([10.0, 3.0]))
        self.assertAllClose(grad, gs.array([[1.0, 3.0, 1.0], [1.0, 3.0, 1.0]]))

    @autograd_and_torch_only
    def test_grad_of_trace(self):
        """The gradient of the trace is the identity, at each point of a batch."""
        points = gs.reshape(gs.arange(8.0), (2, 2, 2))
        value, grad = gs.autodiff.value_and_grad(gs.trace, point_ndims=2)(points)

        self.assertAllClose(value, gs.array([3.0, 11.0]))
        self.assertAllClose(grad, gs.stack([gs.eye(2), gs.eye(2)]))

    @autograd_and_torch_only
    def test_grad_of_solve_sylvester(self):
        """Gradients go through the solution of the Sylvester equation.

        With x X + X x = 1 and x^T Y + Y x^T = 1, the gradient of sum(X) is
        -(Y X^T + X^T Y), and Y = X for a symmetric x. At the identity, the
        eigenvalues are repeated.
        """

        def sylvester_sum(x):
            ones = gs.ones(x.shape[-2:])
            return gs.sum(gs.linalg.solve_sylvester(x, x, ones), axis=(-2, -1))

        points = [gs.eye(2), gs.array([[1.0, 0.0], [0.0, 3.0]])]
        expected_grads = [
            -gs.ones((2, 2)),
            -gs.array([[5.0 / 8.0, 1.0 / 3.0], [1.0 / 3.0, 13.0 / 72.0]]),
        ]
        for point, expected_grad in zip(points, expected_grads):
            _, grad = gs.autodiff.value_and_grad(sylvester_sum, point_ndims=2)(point)
            self.assertAllClose(grad, expected_grad)
