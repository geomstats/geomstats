import pytest

import geomstats.backend as gs
from geomstats.exceptions import AutodiffNotImplementedError
from geomstats.test.test_case import TestCase


class NumpyRaisesTestCase(TestCase):
    def test_raises(self, autodiff_func):
        """Check autodiff functions raise with a backend without autodiff.

        Parameters
        ----------
        autodiff_func : callable
            Function of ``gs.autodiff``.
        """
        with pytest.raises(AutodiffNotImplementedError):
            autodiff_func(self.dummy_func)


class NotDifferentiableTestCase(TestCase):
    def test_value_and_grad_raises(self, func, x, point_ndims=2):
        """Check differentiating a function without gradient raises.

        Parameters
        ----------
        func : callable
            Function relying on a backend function without gradient.
        x : array-like
            Input of the function.
        point_ndims : int
            Number of point dimensions of the input.
        """
        with pytest.raises(NotImplementedError):
            gs.autodiff.value_and_grad(func, point_ndims=point_ndims)(x)


class AutodiffTestCase(TestCase):
    def test_value_and_grad(
        self,
        func,
        x,
        expected_value,
        expected_grad,
        y=None,
        argnums=0,
        point_ndims=1,
        atol=gs.atol,
    ):
        """Check value and gradient.

        Parameters
        ----------
        func : callable
            Function to differentiate.
        x : array-like
            First input of the function.
        expected_value : array-like
            Expected value.
        expected_grad : array-like or tuple
            Expected gradient, or tuple of gradients if `argnums` is a tuple.
        y : array-like
            Second input of the function, if any.
        argnums : int or tuple[int]
            Arguments to differentiate with respect to.
        point_ndims : int or tuple[int]
            Number of point dimensions of the inputs.
        atol : float
            Absolute tolerance.
        """
        inputs = (x,) if y is None else (x, y)
        value_and_grad = gs.autodiff.value_and_grad(
            func, argnums=argnums, point_ndims=point_ndims
        )
        value, grad = value_and_grad(*inputs)
        self.assertAllClose(value, expected_value, atol=atol)

        if isinstance(argnums, int):
            self.assertAllClose(grad, expected_grad, atol=atol)
        else:
            self.assertEqual(len(expected_grad), len(argnums))
            for grad_, expected_grad_ in zip(grad, expected_grad):
                self.assertAllClose(grad_, expected_grad_, atol=atol)

    def test_jacobian(self, func, x, expected, point_ndim=1, atol=gs.atol):
        """Check the jacobian.

        Parameters
        ----------
        func : callable
            Function to differentiate.
        x : array-like, shape=[..., *point_shape]
            Point at which the jacobian is computed.
        expected : array-like, shape=[..., *out_shape, *point_shape]
            Expected jacobian.
        point_ndim : int
            Number of axes of a point.
        atol : float
            Absolute tolerance.
        """
        jac = gs.autodiff.jacobian(func, point_ndim=point_ndim)(x)
        self.assertAllClose(jac, expected, atol=atol)

    def test_hessian(
        self, func, x, expected, point_ndim=1, func_out_ndim=None, atol=gs.atol
    ):
        """Check the hessian.

        Parameters
        ----------
        func : callable
            Function to differentiate.
        x : array-like, shape=[..., *point_shape]
            Point at which the hessian is computed.
        expected : array-like, shape=[..., *out_shape, *point_shape, *point_shape]
            Expected hessian.
        point_ndim : int
            Number of axes of a point.
        func_out_ndim : int
            Number of output components of a vector-valued function.
        atol : float
            Absolute tolerance.
        """
        hess = gs.autodiff.hessian(
            func, point_ndim=point_ndim, func_out_ndim=func_out_ndim
        )(x)
        self.assertAllClose(hess, expected, atol=atol)

    def test_value_and_jacobian(
        self, func, x, expected_value, expected_jacobian, point_ndim=1, atol=gs.atol
    ):
        """Check the value and the jacobian.

        Parameters
        ----------
        func : callable
            Function to differentiate.
        x : array-like, shape=[..., *point_shape]
            Point at which the jacobian is computed.
        expected_value : array-like, shape=[..., *out_shape]
            Expected value.
        expected_jacobian : array-like, shape=[..., *out_shape, *point_shape]
            Expected jacobian.
        point_ndim : int
            Number of axes of a point.
        atol : float
            Absolute tolerance.
        """
        value, jac = gs.autodiff.value_and_jacobian(func, point_ndim=point_ndim)(x)
        self.assertAllClose(value, expected_value, atol=atol)
        self.assertAllClose(jac, expected_jacobian, atol=atol)

    def test_jacobian_and_hessian(
        self,
        func,
        x,
        expected_jacobian,
        expected_hessian,
        point_ndim=1,
        func_out_ndim=None,
        atol=gs.atol,
    ):
        """Check the jacobian and the hessian.

        Parameters
        ----------
        func : callable
            Function to differentiate.
        x : array-like, shape=[..., *point_shape]
            Point at which the derivatives are computed.
        expected_jacobian : array-like, shape=[..., *out_shape, *point_shape]
            Expected jacobian.
        expected_hessian : array-like
            Expected hessian, of shape
            ``[..., *out_shape, *point_shape, *point_shape]``.
        point_ndim : int
            Number of axes of a point.
        func_out_ndim : int
            Number of output components of a vector-valued function.
        atol : float
            Absolute tolerance.
        """
        jac, hess = gs.autodiff.jacobian_and_hessian(
            func, point_ndim=point_ndim, func_out_ndim=func_out_ndim
        )(x)
        self.assertAllClose(jac, expected_jacobian, atol=atol)
        self.assertAllClose(hess, expected_hessian, atol=atol)

    def test_value_jacobian_and_hessian(
        self,
        func,
        x,
        expected_value,
        expected_jacobian,
        expected_hessian,
        point_ndim=1,
        func_out_ndim=None,
        atol=gs.atol,
    ):
        """Check the value, the jacobian and the hessian.

        Parameters
        ----------
        func : callable
            Function to differentiate.
        x : array-like, shape=[..., *point_shape]
            Point at which the derivatives are computed.
        expected_value : array-like, shape=[..., *out_shape]
            Expected value.
        expected_jacobian : array-like, shape=[..., *out_shape, *point_shape]
            Expected jacobian.
        expected_hessian : array-like
            Expected hessian, of shape
            ``[..., *out_shape, *point_shape, *point_shape]``.
        point_ndim : int
            Number of axes of a point.
        func_out_ndim : int
            Number of output components of a vector-valued function.
        atol : float
            Absolute tolerance.
        """
        value, jac, hess = gs.autodiff.value_jacobian_and_hessian(
            func, point_ndim=point_ndim, func_out_ndim=func_out_ndim
        )(x)
        self.assertAllClose(value, expected_value, atol=atol)
        self.assertAllClose(jac, expected_jacobian, atol=atol)
        self.assertAllClose(hess, expected_hessian, atol=atol)

    def test_value_and_jacobian_is_consistent(
        self, func, x, point_ndim=1, atol=gs.atol
    ):
        """Check the value and jacobian against their separate computation.

        Parameters
        ----------
        func : callable
            Function to differentiate, vectorized.
        x : array-like, shape=[..., *point_shape]
            Point at which the jacobian is computed.
        point_ndim : int
            Number of axes of a point.
        atol : float
            Absolute tolerance.
        """
        self.test_value_and_jacobian(
            func,
            x,
            expected_value=func(x),
            expected_jacobian=gs.autodiff.jacobian(func, point_ndim=point_ndim)(x),
            point_ndim=point_ndim,
            atol=atol,
        )

    def test_jacobian_and_hessian_is_consistent(
        self, func, x, point_ndim=1, func_out_ndim=None, atol=gs.atol
    ):
        """Check the jacobian and hessian against their separate computation.

        Parameters
        ----------
        func : callable
            Function to differentiate.
        x : array-like, shape=[..., *point_shape]
            Point at which the derivatives are computed.
        point_ndim : int
            Number of axes of a point.
        func_out_ndim : int
            Number of output components of a vector-valued function.
        atol : float
            Absolute tolerance.
        """
        self.test_jacobian_and_hessian(
            func,
            x,
            expected_jacobian=gs.autodiff.jacobian(func, point_ndim=point_ndim)(x),
            expected_hessian=gs.autodiff.hessian(
                func, point_ndim=point_ndim, func_out_ndim=func_out_ndim
            )(x),
            point_ndim=point_ndim,
            func_out_ndim=func_out_ndim,
            atol=atol,
        )

    def test_value_jacobian_and_hessian_is_consistent(
        self, func, x, point_ndim=1, func_out_ndim=None, atol=gs.atol
    ):
        """Check the value, jacobian and hessian against their separate computation.

        Parameters
        ----------
        func : callable
            Function to differentiate, vectorized.
        x : array-like, shape=[..., *point_shape]
            Point at which the derivatives are computed.
        point_ndim : int
            Number of axes of a point.
        func_out_ndim : int
            Number of output components of a vector-valued function.
        atol : float
            Absolute tolerance.
        """
        self.test_value_jacobian_and_hessian(
            func,
            x,
            expected_value=func(x),
            expected_jacobian=gs.autodiff.jacobian(func, point_ndim=point_ndim)(x),
            expected_hessian=gs.autodiff.hessian(
                func, point_ndim=point_ndim, func_out_ndim=func_out_ndim
            )(x),
            point_ndim=point_ndim,
            func_out_ndim=func_out_ndim,
            atol=atol,
        )
