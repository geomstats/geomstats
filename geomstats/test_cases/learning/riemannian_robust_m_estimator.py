import pytest

import geomstats.backend as gs
from geomstats.learning.riemannian_robust_m_estimator import (
    GradientDescent,
    _scalarmul,
    _scalarmulsum,
    riemannian_variance,
)
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
    MeanEstimatorMixinsTestCase,
)
from geomstats.vectorization import repeat_point


class AutoGradientDescentOneStepTestCase(
    MeanEstimatorMixinsTestCase,
    BaseEstimatorTestCase
):
    """Test onestep autograd quality case"""

    @pytest.mark.random
    def test_onestep_auto_grad_descent_same_as_explicit_grad_descent(self, atol):
        """Test onestep autograd quality case"""
        X = self.data_generator.random_point(n_points=10)

        GD = GradientDescent()
        base1 = GD._set_init_point(
            self.estimator.space, X, init_point_method="mean-projection"
        )
        c = 0.8

        loss_auto_func = self.estimator._set_m_estimator_loss()
        loss_with_base = lambda base: loss_auto_func(
            points=X, base=base, critical_value=c, loss_and_grad=False
        )
        _, grad_auto = gs.autodiff.value_and_grad(
            loss_with_base, point_ndims=self.estimator.space.point_ndim
        )(base1)
        grad_auto = self.estimator.space.to_tangent(grad_auto, base1)

        loss_explicit_func = self.estimator._set_m_estimator_loss()
        _, grad_explicit = loss_explicit_func(
            X, base1, critical_value=c, loss_and_grad=True
        )

        k_multiply = (grad_auto / grad_explicit).reshape(-1)
        print(k_multiply)
        result = gs.unique(
            gs.array([gs.floor(k * 1e5 + 0.5) / 1e5
                      for k in k_multiply if not gs.isnan(k)])
        )
        assert len(result) == 1


class AutoGradientDescentResultTestCase(
    MeanEstimatorMixinsTestCase,
    BaseEstimatorTestCase
):
    """Test autograd quality case"""

    @pytest.mark.random
    def test_auto_grad_descent_result_same_as_explicit_grad_descent(self, atol):
        """Test autograd quality case"""
        X = self.data_generator.random_point(n_points=10)

        res_autograd = self.estimator.fit(X).estimate_.x
        res_explicit = self.estimator_explicit.fit(X).estimate_.x
        res_diff = gs.abs(res_autograd - res_explicit)

        self.assertAllClose(res_diff, gs.zeros(res_diff.shape), atol=atol*10)


class LimitingCofHuberLossTestCase(MeanEstimatorMixinsTestCase, BaseEstimatorTestCase):
    """Test huber limiting case"""

    @pytest.mark.random
    def test_limiting_c_huber_loss(self, atol):
        """Test huber limiting case"""
        X = self.data_generator.random_point(n_points=30)

        huber_mean_0 = self.estimator.fit(X).estimate_.x
        geometric_median = self.estimator_geometric_median.fit(X).estimate_

        huber_mean_inf = self.estimator_inf.fit(X).estimate_.x
        frechet_mean = self.estimator_frechet_mean.fit(X).estimate_

        gm_close = gs.abs(huber_mean_0 - geometric_median)
        fm_close = gs.abs(huber_mean_inf - frechet_mean)

        res = gs.mean(gm_close) + gs.mean(fm_close)

        self.assertAllClose(res, gs.zeros(res.shape), atol=0.0005)


class VarianceTestCase(TestCase):
    """Test Variance quality case"""

    def setup_method(self):
        """Regenerate Random generator"""
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    def test_variance(self, points, base_point, expected, atol, weights=None):
        """Test Variance quality case"""
        res = riemannian_variance(self.space, points, base_point, weights=weights)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_variance_repeated_is_zero(self, n_samples, atol):
        """Test Variance 0 quality case"""
        base_point = point = self.data_generator.random_point(n_points=1)
        points = repeat_point(point, n_samples)

        self.test_variance(points, base_point, 0.0, atol)


class DiffStartingPointSameResultTestCase(
    MeanEstimatorMixinsTestCase,
    BaseEstimatorTestCase
):
    """Test starting point invariance case"""

    @pytest.mark.random
    def test_diff_starting_point_same_result(self, atol):
        """Test starting point invariance case"""
        X = self.data_generator.random_point(n_points=20)

        mean_mp = self.estimator.fit(X).estimate_.x
        mean_md = self.estimator_md.fit(X).estimate_.x
        mean_f = self.estimator_f.fit(X).estimate_.x

        m1_close = gs.abs(mean_mp - mean_md)
        m2_close = gs.abs(mean_md - mean_f)

        self.assertAllClose(m1_close + m2_close, gs.zeros(mean_mp.shape), atol=atol*25)


class SameMestimatorFunctionGivenByCustomAndExplicitTestCase(
    MeanEstimatorMixinsTestCase,
    BaseEstimatorTestCase
):
    """Test custom function working case"""

    @pytest.mark.random
    def test_same_m_estimator_function_given_by_custom_and_explicit(self, atol):
        """Test custom function working case"""
        X = self.data_generator.random_point(n_points=20)
        self.estimator.fit(X)
        res_o = self.estimator.estimate_.x

        self.estimator_custom.set_loss(custom_riemannian_cauchy_loss_grad_cw)
        self.estimator_custom.fit(X)
        res_cw = self.estimator_custom.estimate_.x

        close1 = gs.abs(res_o - res_cw)

        self.assertAllClose(close1, gs.zeros(res_o.shape), atol=atol)


class MestimatorCustomFunctionDifferentInputArgsTestCase(
    MeanEstimatorMixinsTestCase,
    BaseEstimatorTestCase
):
    """Test custom function input change case"""

    def test_custom_function_different_input_arguments(self, atol):
        """Test custom function input change case"""
        X = self.data_generator.random_point(n_points=20)
        self.estimator.points = X
        self.estimator.weights = None
        self.estimator.set_loss()
        loss_e, _ = self.estimator.loss_with_base(X[0])

        self.estimator_custom2.points = X
        self.estimator_custom2.weights = None
        self.estimator_custom2.set_loss(custom_riemannian_cauchy_loss_grad_cw)
        loss_cw = self.estimator_custom2.loss_with_base(X[0])
        self.estimator_custom2.set_loss(custom_riemannian_cauchy_loss_grad_c)
        loss_c = self.estimator_custom2.loss_with_base(X[0])
        self.estimator_custom2.set_loss(custom_riemannian_cauchy_loss_grad_w)
        loss_w = self.estimator_custom2.loss_with_base(X[0])
        self.estimator_custom2.set_loss(custom_riemannian_cauchy_loss_grad_n)
        loss_n = self.estimator_custom2.loss_with_base(X[0])

        res = gs.abs(
            gs.array([
                loss_e - loss_cw,
                loss_cw - loss_c,
                loss_c - loss_w,
                loss_w - loss_n,
                loss_n - loss_e
            ])
        )

        self.assertAllClose(res, gs.zeros(res.shape), atol=atol)


def cauchy_m_estimator(logs, distances, weights, c):
    """Define Euclidean Cauchy Loss function for comparison."""
    sum_weights = gs.sum(weights)
    loss = c**2 / 2 * gs.log(1 + distances**2 / c**2)
    loss = gs.sum(weights * loss) / sum_weights
    grad = _scalarmul(c**2 / (c**2 + distances**2) , logs)
    grad = -1 * _scalarmulsum(weights, grad) / sum_weights
    return loss, grad


def custom_riemannian_cauchy_loss_grad_cw(
        space,
        points,
        base,
        critical_value=2.3849,
        weights=None,
        loss_and_grad=False
):
    """Compute Riemannian Cauchy loss/gradient."""
    c = critical_value
    weights = gs.ones(points.shape[0])
    logs = space.metric.log(point=points, base_point=base)
    distances = space.metric.norm(logs, base)
    loss, grad = cauchy_m_estimator(logs, distances, weights, c)

    if loss_and_grad:
        return loss, space.to_tangent(grad, base_point=base)
    return loss


def custom_riemannian_cauchy_loss_grad_w(
        space,
        points,
        base,
        weights,
        loss_and_grad=False
):
    """Compute Riemannian Cauchy loss/gradient."""
    c = 1
    weights = gs.ones(points.shape[0])
    logs = space.metric.log(point=points, base_point=base)
    distances = space.metric.norm(logs, base)
    loss, grad = cauchy_m_estimator(logs, distances, weights, c)

    if loss_and_grad:
        return loss, space.to_tangent(grad, base_point=base)
    return loss


def custom_riemannian_cauchy_loss_grad_c(
        space,
        points,
        base,
        critical_value,
        loss_and_grad=False
):
    """Compute Riemannian Cauchy loss/gradient."""
    c = critical_value
    weights = gs.ones(points.shape[0])
    logs = space.metric.log(point=points, base_point=base)
    distances = space.metric.norm(logs, base)
    loss, grad = cauchy_m_estimator(logs, distances, weights, c)

    if loss_and_grad:
        return loss, space.to_tangent(grad, base_point=base)
    return loss


def custom_riemannian_cauchy_loss_grad_n(
        space,
        points,
        base,
        loss_and_grad=False
):
    """Compute Riemannian Cauchy loss/gradient."""
    c = 1
    weights = gs.ones(points.shape[0])
    logs = space.metric.log(point=points, base_point=base)
    distances = space.metric.norm(logs, base)
    loss, grad = cauchy_m_estimator(logs, distances, weights, c)

    if loss_and_grad:
        return loss, space.to_tangent(grad, base_point=base)
    return loss
