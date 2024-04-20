import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad_vec
from scipy.stats import entropy

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.information_geometry.normal import NormalDistributions
from geomstats.information_geometry.fisher_rao_metric import FisherRaoMetric
from geomstats.information_geometry.statistical_metric import (
    AlphaConnection,
    DivergenceConnection,
    DualDivergenceConnection,
    StatisticalMetric,
)
from geomstats.test.test_case import TestCase, autograd_only
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


def trapezoidal(y, x):
    d = np.diff(x)
    return gs.sum((y[0:-1] + y[1:]) * d / 2)

def bregman_divergence(func):
    def _bregman_divergence(point, base_point):
        grad_func = gs.autodiff.value_and_grad(func)
        func_basepoint, grad_func_basepoint = grad_func(base_point)
        bregman_div = (
            func(point)
            - func_basepoint
            - gs.dot(point - base_point, grad_func_basepoint)
        )
        return bregman_div

    return _bregman_divergence

def potential_function(point):
    return gs.sum(point**4)


@autograd_only
# @pytest.mark.slow
class TestStatisticalMetric(TestCase):
    """
    Test StatisticalMetric class by verifying expected induced metric, connection,
    and dual connection properties in the case of Bregman divergence.

    Using properties from Nielson's An Elementary Introduction to
    Information Geometry, Section 3.7 on page 14 (https://arxiv.org/abs/1808.08271)
    """

    def setup_method(self):

        self.potential_function = potential_function
        self.breg_divergence = bregman_divergence(self.potential_function)
        self.euclidean_space = Euclidean(dim=2, equip=False)
        self.bregman_primal_connection = DivergenceConnection(
            space=self.euclidean_space, divergence=self.breg_divergence
        )
        self.bregman_dual_connection = DualDivergenceConnection(
            space=self.euclidean_space, divergence=self.breg_divergence
        )
        self.bregman_alpha_connection = AlphaConnection(
            space=self.euclidean_space,
            alpha=0.0,
            primal_connection=self.bregman_primal_connection,
            dual_connection=self.bregman_dual_connection,
        )
        self.euc_stat_metric = StatisticalMetric(
            space=self.euclidean_space,
            divergence=self.breg_divergence,
            primal_connection=self.bregman_primal_connection,
            dual_connection=self.bregman_dual_connection,
        )
        self.euclidean_space.metric = self.euc_stat_metric
        self.base_point = gs.array([2.1, 2.3])

        self.normal_manifold = NormalDistributions(sample_dim=1)

        def KL_divergence(theta1, theta2):
            def _func_to_integrate(x):
                return self.normal_manifold.point_to_pdf(theta1)(x) * gs.log(
                    (self.normal_manifold.point_to_pdf(theta1)(x))
                    / (self.normal_manifold.point_to_pdf(theta2)(x))
                )

            domain = gs.linspace(-10, 10, 1000)
            return trapezoidal(_func_to_integrate(domain), domain)
        
        self.normal_stat_metric = StatisticalMetric(
            space=self.normal_manifold,
            divergence=KL_divergence,
            primal_connection=DivergenceConnection(
                space=self.normal_manifold, divergence=KL_divergence
            ),
            dual_connection=DualDivergenceConnection(
                space=self.normal_manifold, divergence=KL_divergence
            ),
        )

    def test_KL_metric_matrix(
            self,
        ):
        KL_induced_metric =self.normal_stat_metric.metric_matrix(self.base_point)
        fisher_metric = FisherRaoMetric(space=self.normal_manifold, support=[-10, 10])
        fisher_induced_metric = fisher_metric.metric_matrix(self.base_point)
        self.assertAllClose(KL_induced_metric, fisher_induced_metric, atol=1e-4)

    def test_bregman_metric_matrix(
        self,
    ):
        """Test equation (58) on page 15"""
        potential_hessian = gs.autodiff.hessian(self.potential_function)
        potential_hessian_base_point = potential_hessian(self.base_point)
        metric_matrix_base_point = self.euc_stat_metric.metric_matrix(self.base_point)
        self.assertAllClose(metric_matrix_base_point, potential_hessian_base_point)

    def test_bregman_divergence_christoffels(
        self,
    ):
        """Test equation (59) on page 15"""
        divergence_christoffels_base_point = self.bregman_primal_connection.christoffels(
            self.base_point
        )
        self.assertAllClose(
            divergence_christoffels_base_point,
            gs.zeros(divergence_christoffels_base_point.shape),
        )

    def test_bregman_dual_divergence_christoffels(
        self,
    ):
        pass

    def test_bregman_alpha_christoffels(
        self,
    ):
        """Test that LC connection is recovered"""
        alpha_christoffels_base_point = self.bregman_alpha_connection.christoffels(
            self.base_point
        )
        metric_base_point = self.euclidean_space.metric.metric_matrix(self.base_point)

        first_kind_alpha_christoffels_base_point = gs.einsum(
            "...kij,...km->...mij", alpha_christoffels_base_point, metric_base_point
        )

        levi_civita_christoffels_base_point = self.euclidean_space.metric.christoffels(
            self.base_point
        )

        firsk_kind_levi_civita_christoffels_base_point = gs.einsum(
            "...kij,...km->...mij",
            levi_civita_christoffels_base_point,
            metric_base_point,
        )

        self.assertAllClose(
            first_kind_alpha_christoffels_base_point,
            firsk_kind_levi_civita_christoffels_base_point,
        )

    def test_bregman_amari_divergence_tensor(
        self,
    ):
        """Test equation (60) on page 15"""
        potential_func_first = gs.autodiff.jacobian(self.potential_function)
        potential_func_second = gs.autodiff.jacobian(potential_func_first)
        potential_func_third = gs.autodiff.jacobian(potential_func_second)
        potential_func_third_base_point = potential_func_third(self.base_point)
        amari_divergence_tensor_base_point = (
            self.euc_stat_metric.amari_divergence_tensor(self.base_point)
        )

        self.assertAllClose(
            amari_divergence_tensor_base_point, potential_func_third_base_point
        )
