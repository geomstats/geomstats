"""Unit tests for automatic differentiation in different backends."""

import warnings

import numpy as _np
import pytest

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.grassmannian import Grassmannian
from geomstats.geometry.special_euclidean import SpecialEuclidean


def _sphere_immersion(point):
    radius = 4.0
    theta = point[0]
    phi = point[1]
    x = gs.sin(theta) * gs.cos(phi)
    y = gs.sin(theta) * gs.sin(phi)
    z = gs.cos(theta)
    return gs.array([radius * x, radius * y, radius * z])


def _first_component_of_sphere_immersion(point):
    """First component of the sphere immersion function.

    This returns a vector of dim 1.
    """
    radius = 4.0
    theta = point[0]
    phi = point[1]
    x = gs.sin(theta) * gs.cos(phi)
    return gs.array([radius * x])


def _first_component_of_sphere_immersion_scalar(point):
    """First component of the sphere immersion function.

    This returns a scalar.
    """
    radius = 4.0
    theta = point[0]
    phi = point[1]
    x = gs.sin(theta) * gs.cos(phi)
    return radius * x


def _sphere_immersion_at_phi0_from_scalar_input(theta):
    """Immersion of a one-dim input."""
    radius = 4.0
    x = gs.sin(theta)
    y = 0
    z = gs.cos(theta)
    return gs.array([radius * x, radius * y, radius * z])


def _sphere_immersion_at_phi0_from_one_dim_input(theta):
    """Immersion of a one-dim input."""
    radius = 4.0
    x = gs.sin(theta[0])
    y = 0
    z = gs.cos(theta[0])
    return gs.array([radius * x, radius * y, radius * z])


class TestAutodiff(tests.conftest.TestCase):
    def setup_method(self):
        warnings.simplefilter("ignore", category=ImportWarning)
        self.n_samples = 2

    @tests.conftest.np_only
    def test_value_and_grad_np_backend(self):
        n = 10
        vector = gs.ones(n)
        with pytest.raises(RuntimeError):
            gs.autodiff.value_and_grad(lambda v: gs.sum(v**2))(vector)

    @tests.conftest.autograd_tf_and_torch_only
    def test_value_and_grad_one_vector_var(self):
        n = 10
        vector = gs.ones(n)
        result_loss, result_grad = gs.autodiff.value_and_grad(lambda v: gs.sum(v**2))(
            vector
        )
        expected_loss = n
        expected_grad = 2 * vector

        self.assertAllClose(result_loss, expected_loss)
        self.assertAllClose(result_grad, expected_grad)

    @tests.conftest.autograd_and_tf_only
    def test_value_and_grad_dist(self):
        space = SpecialEuclidean(3)
        metric = space.metric
        point = space.random_point()
        id = space.identity
        result_loss, result_grad = gs.autodiff.value_and_grad(
            lambda v: metric.squared_dist(v, id)
        )(point)

        expected_loss = metric.squared_dist(point, id)
        expected_grad = -2 * metric.log(id, point)

        self.assertAllClose(result_loss, expected_loss)
        self.assertAllClose(result_grad, expected_grad)

    @tests.conftest.autograd_tf_and_torch_only
    def test_value_and_grad_dist_grassmann(self):
        space = Grassmannian(3, 2)
        metric = space.metric
        point = space.random_point()
        vector = space.to_tangent(space.random_point(), point)
        result_loss, result_grad = gs.autodiff.value_and_grad(
            lambda v: metric.squared_norm(v, point)
        )(vector)

        expected_loss = metric.squared_norm(vector, point)
        expected_grad = 2 * vector

        self.assertAllClose(result_loss, expected_loss)
        self.assertAllClose(result_grad, expected_grad)

    @tests.conftest.autograd_tf_and_torch_only
    def test_value_and_grad_one_vector_var_np_input(self):
        n = 10
        vector = _np.ones(n)
        result_loss, result_grad = gs.autodiff.value_and_grad(lambda v: gs.sum(v**2))(
            vector
        )
        expected_loss = n
        expected_grad = 2 * vector
        self.assertAllClose(result_loss, expected_loss)
        self.assertAllClose(result_grad, expected_grad)

    @tests.conftest.autograd_tf_and_torch_only
    def test_value_and_grad_two_scalars_vars(self):
        def func(x, y):
            return gs.sum((x - y) ** 2)

        arg_x = 1.0
        arg_y = 2.0
        val, grad = gs.autodiff.value_and_grad(func)(arg_x, arg_y)

        self.assertAllClose(val, 1.0)
        self.assertTrue(isinstance(grad, tuple))
        self.assertAllClose(grad[0], -2)
        self.assertAllClose(grad[1], 2.0)

    @tests.conftest.autograd_tf_and_torch_only
    def test_value_and_grad_two_vectors_vars(self):
        def func(x, y):
            return gs.sum((x - y) ** 2)

        arg_x = gs.array([1.0, 2.0])
        arg_y = gs.array([2.0, 3.0])
        val, grad = gs.autodiff.value_and_grad(func)(arg_x, arg_y)

        self.assertAllClose(val, 2.0)
        self.assertTrue(isinstance(grad, tuple))
        self.assertAllClose(grad[0], gs.array([-2.0, -2.0]))
        self.assertAllClose(grad[1], gs.array([2.0, 2.0]))

    @tests.conftest.autograd_tf_and_torch_only
    def test_value_and_grad_two_matrix_vars(self):
        def func(x, y):
            return gs.sum((x - y) ** 2)

        arg_x = gs.array([[1.0, 3.0], [2.0, 3.0]])
        arg_y = gs.array([[2.0, 5.0], [0.0, 4.0]])
        val, grad = gs.autodiff.value_and_grad(func)(arg_x, arg_y)
        self.assertAllClose(val, 10.0)
        self.assertAllClose(grad[0], gs.array([[-2.0, -4.0], [4.0, -2.0]]))
        self.assertAllClose(grad[1], gs.array([[2.0, 4.0], [-4.0, 2.0]]))

    @tests.conftest.autograd_tf_and_torch_only
    def test_custom_gradient_one_vector_var(self):
        """Assign made-up gradient to test custom_gradient."""

        def grad_x(x):
            return 6 * x

        @gs.autodiff.custom_gradient(grad_x)
        def func(x):
            return gs.sum(x**2)

        arg_x = gs.array([1.0, 3.0])
        result_value, result_grad = gs.autodiff.value_and_grad(func)(arg_x)

        expected_value = 10.0
        expected_grad = gs.array([6.0, 18.0])

        self.assertAllClose(result_value, expected_value)
        self.assertAllClose(result_grad, expected_grad)

    @tests.conftest.autograd_tf_and_torch_only
    def test_custom_gradient_two_vector_vars(self):
        """Assign made-up gradient to test custom_gradient."""

        def grad_x(x, y):
            return 6 * (x - y)

        def grad_y(x, y):
            return 6 * (y - x)

        @gs.autodiff.custom_gradient(grad_x, grad_y)
        def func(x, y):
            return gs.sum((x - y) ** 2)

        arg_x = gs.array([1.0, 3.0])
        arg_y = gs.array([2.0, 5.0])

        result_val, result_grad = gs.autodiff.value_and_grad(func)(arg_x, arg_y)

        self.assertTrue(isinstance(result_grad, tuple))
        result_grad_x, result_grad_y = result_grad

        expected_val = func(arg_x, arg_y)
        expected_grad_x = grad_x(arg_x, arg_y)
        expected_grad_y = grad_y(arg_x, arg_y)

        self.assertAllClose(result_val, expected_val)
        self.assertAllClose(result_grad_x, expected_grad_x)
        self.assertAllClose(result_grad_y, expected_grad_y)

    @tests.conftest.autograd_tf_and_torch_only
    def test_custom_gradient_two_matrix_vars(self):
        """Assign made-up gradient to test custom_gradient."""

        def grad_x(x, y):
            return 6 * (x - y)

        def grad_y(x, y):
            return 6 * (y - x)

        @gs.autodiff.custom_gradient(grad_x, grad_y)
        def func(x, y):
            return gs.sum((x - y) ** 2)

        arg_x = gs.array([[1.0, 3.0], [2.0, 3.0]])
        arg_y = gs.array([[2.0, 5.0], [0.0, 4.0]])

        result_val, result_grad = gs.autodiff.value_and_grad(func)(arg_x, arg_y)

        self.assertTrue(isinstance(result_grad, tuple))
        result_grad_x, result_grad_y = result_grad

        expected_val = func(arg_x, arg_y)
        expected_grad_x = grad_x(arg_x, arg_y)
        expected_grad_y = grad_y(arg_x, arg_y)

        self.assertAllClose(result_val, expected_val)
        self.assertAllClose(result_grad_x, expected_grad_x)
        self.assertAllClose(result_grad_y, expected_grad_y)

    @tests.conftest.autograd_tf_and_torch_only
    def test_custom_gradient_composed_two_matrix_vars(self):
        """Assign made-up gradient to test custom_gradient."""

        def grad_x(x, y):
            return 6 * (x - y)

        def grad_y(x, y):
            return 6 * (y - x)

        @gs.autodiff.custom_gradient(grad_x, grad_y)
        def func(x, y):
            return gs.sum((x - y) ** 2)

        arg_x = gs.array([[1.0, 3.0], [2.0, 3.0]])
        const_y = gs.array([[2.0, 5.0], [0.0, 4.0]])

        def func_2(x):
            return gs.exp(-0.5 * func(x, const_y))

        result_value, result_grad = gs.autodiff.value_and_grad(func_2)(arg_x)
        expected_value = func_2(arg_x)
        expected_grad = 3 * (const_y - arg_x) * expected_value
        self.assertAllClose(result_value, expected_value)
        self.assertAllClose(result_grad, expected_grad)

    @tests.conftest.autograd_tf_and_torch_only
    def test_custom_gradient_composed_with_dummy_two_matrix_vars(self):
        """Assign made-up gradient to test custom_gradient."""

        def grad_dummy(dummy, x, y):
            return dummy

        def grad_x(dummy, x, y):
            return 6 * dummy * (x - y)

        def grad_y(dummy, x, y):
            return 6 * dummy * (y - x)

        @gs.autodiff.custom_gradient(grad_dummy, grad_x, grad_y)
        def func(dummy, x, y):
            return dummy * gs.sum((x - y) ** 2)

        const_y = gs.array([[2.0, 5.0], [0.0, 4.0]])
        const_dummy = gs.array(4.0)

        def func_of_x(x):
            return func(const_dummy, x, const_y)

        arg_x = gs.array([[1.0, 3.0], [2.0, 3.0]])
        result_value, result_grad = gs.autodiff.value_and_grad(func_of_x)(arg_x)
        expected_value = func_of_x(arg_x)
        expected_grad = grad_x(const_dummy, arg_x, const_y)

        self.assertAllClose(result_value, expected_value)
        self.assertAllClose(result_grad, expected_grad)

    @tests.conftest.autograd_tf_and_torch_only
    def test_custom_gradient_chain_rule_one_scalar_var(self):
        """Assign made-up gradient to test custom_gradient."""

        def fun1_grad(x):
            return gs.array(3.0, dtype=x.dtype)

        @gs.autodiff.custom_gradient(fun1_grad)
        def fun1(x):
            return x

        def fun2(x):
            out = fun1(x) ** 2
            return out

        def fun2_grad(x):
            return 2 * x

        arg = gs.array(10.0)

        result_value, result_grad = gs.autodiff.value_and_grad(fun2)(arg)
        expected_value = fun2(arg)
        expected_grad_explicit = 60.0

        expected_grad_chain_rule = fun2_grad(fun1(arg)) * fun1_grad(arg)

        self.assertAllClose(result_value, expected_value)
        self.assertAllClose(result_grad, expected_grad_explicit)
        self.assertAllClose(result_grad, expected_grad_chain_rule)

    @tests.conftest.autograd_tf_and_torch_only
    def test_custom_gradient_chain_rule_one_vector_var(self):
        def fun1_grad(x):
            return 6 * x

        @gs.autodiff.custom_gradient(fun1_grad)
        def fun1(x):
            return gs.sum(x) ** 2

        def fun2(x):
            out = fun1(x) ** 3
            return out

        def fun2_grad(x):
            return 3 * x**2

        arg = gs.array([1.0, 2.0])

        result_value, result_grad = gs.autodiff.value_and_grad(fun2)(arg)

        expected_value = fun2(arg)
        expected_grad_explicit = 18 * fun1(arg) ** 2 * arg
        expected_grad_chain_rule = fun2_grad(fun1(arg)) * fun1_grad(arg)

        self.assertAllClose(result_value, expected_value)
        self.assertAllClose(result_grad, expected_grad_explicit)
        self.assertAllClose(result_grad, expected_grad_chain_rule)

    @tests.conftest.autograd_and_tf_only
    def test_custom_gradient_squared_dist(self):
        def squared_dist_grad_a(point_a, point_b, metric):
            return -2 * metric.log(point_b, point_a)

        def squared_dist_grad_b(point_a, point_b, metric):
            return -2 * metric.log(point_a, point_b)

        @gs.autodiff.custom_gradient(squared_dist_grad_a, squared_dist_grad_b)
        def squared_dist(point_a, point_b, metric):
            dist = metric.squared_dist(point_a, point_b)
            return dist

        space = SpecialEuclidean(n=2)
        const_metric = space.left_canonical_metric
        const_point_b = space.random_point()

        def func(x):
            return squared_dist(x, const_point_b, metric=const_metric)

        arg_point_a = space.random_point()
        expected_value = func(arg_point_a)
        expected_grad = -2 * const_metric.log(const_point_b, arg_point_a)
        result_value, result_grad = gs.autodiff.value_and_grad(func)(arg_point_a)
        self.assertAllClose(result_value, expected_value)
        self.assertAllClose(result_grad, expected_grad)

    @tests.conftest.autograd_and_tf_only
    def test_custom_gradient_in_action(self):
        space = SpecialEuclidean(n=2)
        const_metric = space.left_canonical_metric
        const_point_b = space.random_point()

        def func(x):
            return const_metric.squared_dist(x, const_point_b)

        arg_point_a = space.random_point()
        func_with_grad = gs.autodiff.value_and_grad(func)
        result_value, result_grad = func_with_grad(arg_point_a)
        expected_value = const_metric.squared_dist(arg_point_a, const_point_b)
        expected_grad = -2 * const_metric.log(const_point_b, arg_point_a)

        self.assertAllClose(result_value, expected_value)
        self.assertAllClose(result_grad, expected_grad)

        loss, grad = func_with_grad(const_point_b)
        self.assertAllClose(loss, 0.0)
        self.assertAllClose(grad, gs.zeros_like(grad))

    @tests.conftest.autograd_tf_and_torch_only
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

    @tests.conftest.autograd_tf_and_torch_only
    def test_jacobian_of_scalar_function(self):
        """Test that jacobians are consistent across backends.

        The jacobian of a function f going from an input space A to an output
        space B is a matrix of shape (dim_B, dim_A).
        - The columns index the derivatives wrt. the coordinates of the input space A.
        - The rows index the coordinates of the output space B.
        """
        radius = 4.0
        dim = 2

        point = gs.array([gs.pi / 3, gs.pi])
        theta = point[0]
        phi = point[1]
        jacobian_1i = gs.autodiff.jacobian(_first_component_of_sphere_immersion)(point)

        expected_1i = gs.array(
            [
                [
                    radius * gs.cos(theta) * gs.cos(phi),
                    -radius * gs.sin(theta) * gs.sin(phi),
                ]
            ]
        )

        self.assertAllClose(jacobian_1i.shape, (1, dim))
        self.assertAllClose(jacobian_1i, expected_1i)

        jacobian_1i = gs.autodiff.jacobian(_first_component_of_sphere_immersion_scalar)(
            point
        )

        expected_1i = gs.array(
            [
                radius * gs.cos(theta) * gs.cos(phi),
                -radius * gs.sin(theta) * gs.sin(phi),
            ]
        )

        self.assertAllClose(jacobian_1i.shape, (dim,))
        self.assertAllClose(jacobian_1i, expected_1i)

    @tests.conftest.autograd_tf_and_torch_only
    def test_jacobian_of_scalar_input(self):
        radius = 4.0
        embedding_dim = 3

        point = gs.array(gs.pi / 3)
        jacobian_1i = gs.autodiff.jacobian(_sphere_immersion_at_phi0_from_scalar_input)(
            point
        )

        expected = gs.array(
            [
                radius * gs.cos(point),
                0,
                -radius * gs.sin(point),
            ]
        )

        self.assertAllClose(jacobian_1i.shape, (embedding_dim,))
        self.assertAllClose(jacobian_1i, expected)

    @tests.conftest.autograd_tf_and_torch_only
    def test_jacobian_of_one_dim_input(self):
        radius = 4.0
        embedding_dim = 3

        point = gs.array([gs.pi / 3])
        jacobian_1i = gs.autodiff.jacobian(
            _sphere_immersion_at_phi0_from_one_dim_input
        )(point)

        expected = gs.array(
            [
                [radius * gs.cos(point[0])],
                [0],
                [-radius * gs.sin(point[0])],
            ]
        )

        self.assertAllClose(jacobian_1i.shape, (embedding_dim, 1))
        self.assertAllClose(jacobian_1i, expected)

    @tests.conftest.autograd_tf_and_torch_only
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

    @tests.conftest.autograd_tf_and_torch_only
    def test_jacobian_vec_of_scalar_function(self):
        """Test that jacobian_vec is correctly vectorized.

        The autodiff jacobian is not vectorized by default in torch, tf and autograd.

        The jacobian of a function f going from an input space A to an output
        space B is a matrix of shape (dim_B, dim_A).
        - The columns index the derivatives wrt. the coordinates of the input space A.
        - The rows index the coordinates of the output space B.
        """
        radius = 4.0
        dim = 2

        points = gs.array([[gs.pi / 3, gs.pi], [gs.pi / 5, gs.pi / 2]])
        thetas = points[:, 0]
        phis = points[:, 1]
        jacobian_1i = gs.autodiff.jacobian_vec(_first_component_of_sphere_immersion)(
            points
        )

        expected_1i = gs.stack(
            [
                gs.array(
                    [
                        [
                            radius * gs.cos(theta) * gs.cos(phi),
                            -radius * gs.sin(theta) * gs.sin(phi),
                        ]
                    ]
                )
                for theta, phi in zip(thetas, phis)
            ]
        )

        self.assertAllClose(jacobian_1i.shape, (len(points), 1, dim))
        self.assertAllClose(jacobian_1i.shape, expected_1i.shape)
        self.assertAllClose(jacobian_1i, expected_1i)

        jacobian_1i = gs.autodiff.jacobian_vec(
            _first_component_of_sphere_immersion_scalar
        )(points)

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

        self.assertAllClose(jacobian_1i.shape, (len(points), dim))
        self.assertAllClose(jacobian_1i.shape, expected_1i.shape)
        self.assertAllClose(jacobian_1i, expected_1i)

    @tests.conftest.autograd_tf_and_torch_only
    def test_jacobian_vec_of_scalar_input(self):
        radius = 4.0
        embedding_dim = 3

        point = gs.array([gs.pi / 3, gs.pi / 6])
        jacobian_1i = gs.autodiff.jacobian(_sphere_immersion_at_phi0_from_scalar_input)(
            point
        )

        expected = gs.array(
            [
                radius * gs.cos(point[0]),
                0,
                -radius * gs.sin(point[0]),
            ],
            [
                radius * gs.cos(point[1]),
                0,
                -radius * gs.sin(point[1]),
            ],
        )

        self.assertAllClose(
            jacobian_1i.shape,
            (
                len(point),
                embedding_dim,
            ),
        )
        self.assertAllClose(jacobian_1i, expected)

    @tests.conftest.autograd_tf_and_torch_only
    def test_jacobian_vec_of_one_dim_input(self):
        radius = 4.0
        embedding_dim = 3

        point = gs.array([[gs.pi / 3], [gs.pi / 6]])
        jacobian_1i = gs.autodiff.jacobian(
            _sphere_immersion_at_phi0_from_one_dim_input
        )(point)

        expected = gs.array(
            [
                [radius * gs.cos(point[0, 0])],
                [0],
                [-radius * gs.sin(point[0, 0])],
            ],
            [
                [radius * gs.cos(point[1, 0])],
                [0],
                [-radius * gs.sin(point[1, 0])],
            ],
        )

        self.assertAllClose(jacobian_1i.shape, (len(point), embedding_dim, 1))
        self.assertAllClose(jacobian_1i, expected)

    @tests.conftest.autograd_tf_and_torch_only
    def test_hessian(self):
        radius = 4.0
        dim = 2

        point = gs.array([gs.pi / 3, gs.pi])
        theta = point[0]
        phi = point[1]
        hessian_1ij = gs.autodiff.hessian(_first_component_of_sphere_immersion)(point)

        expected_1ij = radius * gs.array(
            [
                [-gs.sin(theta) * gs.cos(phi), -gs.cos(theta) * gs.sin(phi)],
                [-gs.cos(theta) * gs.sin(phi), -gs.sin(theta) * gs.cos(phi)],
            ]
        )

        self.assertAllClose(hessian_1ij.shape, (dim, dim))
        self.assertAllClose(hessian_1ij.shape, expected_1ij.shape)
        self.assertAllClose(hessian_1ij, expected_1ij)

    @tests.conftest.autograd_tf_and_torch_only
    def test_hessian_vec(self):
        """Hessian is not vectorized by default in torch, tf and autograd."""
        radius = 4.0
        dim = 2

        points = gs.array([[gs.pi / 3, gs.pi], [gs.pi / 4, gs.pi / 2]])
        thetas = points[:, 0]
        phis = points[:, 1]
        hessian_1ij = gs.autodiff.hessian_vec(_first_component_of_sphere_immersion)(
            points
        )

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
