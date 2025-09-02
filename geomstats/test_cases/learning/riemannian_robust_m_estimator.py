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


class HuberMeanExtremeCTestCase(MeanEstimatorMixinsTestCase, BaseEstimatorTestCase):
    @pytest.mark.random
    def test_huber_extreme_c(self, atol):
        X = self.data_generator.random_point(n_points=30)

        huber_mean_0 = self.estimator.fit(X).estimate_.x
        geometric_median = self.estimator_geometric_median.fit(X).estimate_

        huber_mean_inf = self.estimator_inf.fit(X).estimate_.x
        frechet_mean = self.estimator_frechet_mean.fit(X).estimate_

        gm_close = gs.abs(huber_mean_0 - geometric_median)
        fm_close = gs.abs(huber_mean_inf - frechet_mean)

        assert gs.mean(gm_close) + gs.mean(fm_close) < 0.0005


class AutoGradientDescentTestCase(MeanEstimatorMixinsTestCase, BaseEstimatorTestCase):
    @pytest.mark.random
    def test_auto_grad_descent_same_as_explicit_grad_descent(self, atol):
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
        grad_auto = -1 * self.estimator.space.to_tangent(grad_auto, base1)

        loss_explicit_func = self.estimator._set_m_estimator_loss()
        _, grad_explicit = loss_explicit_func(
            X, base1, critical_value=c, loss_and_grad=True
        )

        k_multiply = (grad_auto / grad_explicit).reshape(-1)
        print(k_multiply)
        result = gs.unique(
            gs.array([gs.floor(k * 1e5 + 0.5) / 1e5 \
                      for k in k_multiply if str(k) not in ["nan", "inf"]])
        )
        assert len(result) == 1


class VarianceTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    def test_variance(self, points, base_point, expected, atol, weights=None):
        res = riemannian_variance(self.space, points, base_point, weights=weights)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_variance_repeated_is_zero(self, n_samples, atol):
        base_point = point = self.data_generator.random_point(n_points=1)
        points = repeat_point(point, n_samples)

        self.test_variance(points, base_point, 0.0, atol)


class DiffStartingPointSameResultTestCase(
    MeanEstimatorMixinsTestCase, 
    BaseEstimatorTestCase):
    @pytest.mark.random
    def test_diff_starting_point_same_result(self, atol):
        X = self.data_generator.random_point(n_points=20)

        mean_mp = self.estimator.fit(X).estimate_.x
        mean_md = self.estimator_md.fit(X).estimate_.x
        mean_f = self.estimator_f.fit(X).estimate_.x

        m1_close = gs.abs(mean_mp - mean_md)
        m2_close = gs.abs(mean_md - mean_f)

        self.assertAllClose(m1_close + m2_close, gs.zeros(mean_mp.shape), atol=atol*1e2)


class SameMestimatorFunctionGivenByCustomAndExplicitTestCase(
    MeanEstimatorMixinsTestCase, 
    BaseEstimatorTestCase):
    @pytest.mark.random
    def test_same_m_estimator_function_given_by_custom_and_explicit(self, atol):
        X = self.data_generator.random_point(n_points=20)
        self.estimator.fit(X)
        res_o = self.estimator.estimate_.x

        self.estimator_custom._set_loss(custom_riemannian_cauchy_loss_grad_cw)
        self.estimator_custom.fit(X)
        res_cw = self.estimator_custom.estimate_.x

        close1 = gs.abs(res_o - res_cw)

        self.assertAllClose(close1, gs.zeros(res_o.shape), atol=atol)


def cauchy_m_estimator(logs, distances, weights, c):
    """Define Euclidean Cauchy Loss function for comparison."""
    sum_weights = gs.sum(weights)
    loss = c**2 / 2 * gs.log(1 + distances**2 / c**2)
    loss = gs.sum(weights * loss) / sum_weights
    grad = _scalarmul(c**2 / (c**2 + distances**2) , logs)
    grad = _scalarmulsum(weights, grad) / sum_weights
    return loss, grad


def custom_riemannian_cauchy_loss_grad_cw(
        space, 
        points, 
        base, 
        critical_value=2.3849, 
        weights=None, 
        loss_and_grad=False):
    """Compute Riemannian Cauchy loss/gradient."""
    c = critical_value
    weights = gs.ones(points.shape[0])
    logs = space.metric.log(point=points, base_point=base)
    distances = space.metric.norm(logs, base)
    loss, grad = cauchy_m_estimator(logs, distances, weights, c)

    if loss_and_grad:
        return loss, space.to_tangent(grad, base_point=base)
    return loss
