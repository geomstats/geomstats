"""Unit tests for automatic differentiation in different backends."""

import warnings

import numpy as _np
import pytest

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.grassmannian import Grassmannian
from geomstats.geometry.special_euclidean import SpecialEuclidean


class TestAutodiff(geomstats.tests.TestCase):
    def setup_method(self):
        warnings.simplefilter("ignore", category=ImportWarning)
        self.n_samples = 2

    @geomstats.tests.np_only
    def test_value_and_grad_np_backend(self):
        n = 10
        vector = gs.ones(n)
        with pytest.raises(RuntimeError):
            gs.autodiff.value_and_grad(lambda v: gs.sum(v**2))(vector)

    @geomstats.tests.autograd_tf_and_torch_only
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

    @geomstats.tests.autograd_and_tf_only
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

    @geomstats.tests.autograd_tf_and_torch_only
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

    @geomstats.tests.autograd_tf_and_torch_only
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

    @geomstats.tests.autograd_tf_and_torch_only
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

    @geomstats.tests.autograd_tf_and_torch_only
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

    @geomstats.tests.autograd_tf_and_torch_only
    def test_value_and_grad_two_matrix_vars(self):
        def func(x, y):
            return gs.sum((x - y) ** 2)

        arg_x = gs.array([[1.0, 3.0], [2.0, 3.0]])
        arg_y = gs.array([[2.0, 5.0], [0.0, 4.0]])
        val, grad = gs.autodiff.value_and_grad(func)(arg_x, arg_y)
        self.assertAllClose(val, 10.0)
        self.assertAllClose(grad[0], gs.array([[-2.0, -4.0], [4.0, -2.0]]))
        self.assertAllClose(grad[1], gs.array([[2.0, 4.0], [-4.0, 2.0]]))

    @geomstats.tests.autograd_tf_and_torch_only
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

    @geomstats.tests.autograd_tf_and_torch_only
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

    @geomstats.tests.autograd_tf_and_torch_only
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

    @geomstats.tests.autograd_tf_and_torch_only
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

    @geomstats.tests.autograd_tf_and_torch_only
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

    @geomstats.tests.autograd_tf_and_torch_only
    def test_custom_gradient_chain_rule_one_scalar_var(self):
        """Assign made-up gradient to test custom_gradient."""

        def fun1_grad(x):
            return 3.0

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

    @geomstats.tests.autograd_tf_and_torch_only
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

    @geomstats.tests.autograd_and_tf_only
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

    @geomstats.tests.autograd_and_tf_only
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
