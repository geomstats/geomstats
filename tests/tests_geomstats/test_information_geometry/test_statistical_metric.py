import pytest

import geomstats.backend as gs
from geomstats.information_geometry.statistical_metric import StatisticalMetric
from geomstats.test.test_case import TestCase, autograd_and_torch_only

def bregman_divergence(func):
    def _bregman_divergence(point, base_point):
        grad_func = gs.autodiff.value_and_grad(func)
        func_basepoint, grad_func_basepoint = grad_func(base_point)
        bregman_div = func(point) - func_basepoint - gs.dot(point - base_point, grad_func_basepoint)
        return bregman_div
    return _bregman_divergence 

def potential_function(point):
    return gs.sum(point**4)


breg_divergence = bregman_divergence(potential_function)


@autograd_and_torch_only
@pytest.mark.slow
class TestStatisticalMetric(TestCase):
    """
    Test StatisticalMetric class by verifying expected induced metric, connection, and dual connection properties in the case of Bregman divergence.

    Using properties from Nielson's An Elementary Introduction to Information Geometry Section 3.7 (page 14)
    """

    def setup_method(self):
        # def bregman_divergence(func):
        #     def _bregman_divergence(point, base_point):
        #         grad_func = gs.autodiff.value_and_grad(func)
        #         func_basepoint, grad_func_basepoint = grad_func(base_point)
        #         bregman_div = func(point) - func_basepoint - gs.dot(point - base_point, grad_func_basepoint)
        #         return bregman_div
        #     return _bregman_divergence 
        
        # def potential_function(point):
        #     return gs.sum(point**4)

        # self.potential_function = potential_function
        # self.breg_divergence = bregman_divergence(potential_function)
        self.stat_metric = StatisticalMetric(dim=2, divergence=breg_divergence)
        # self.potential_function = potential_function()
        # self.bregman_divergence = bregman_divergence()
        self.base_point = gs.random.uniform(low=-10, high=10, size=(2,))

        print(self.stat_metric.metric_matrix(self.base_point))

    # def potential_function(self, point):
    #     return gs.sum(point**4)

    def test_metric_matrix(self,):
        """Test equation (58)"""

        potential_hessian = gs.autodiff.hessian(potential_function)
        potential_hessian_base_point = potential_hessian(self.base_point)
        metric_matrix_base_point = self.stat_metric.metric_matrix(self.base_point)
        self.assertAllClose(-1*metric_matrix_base_point, potential_hessian_base_point)

    def test_divergence_christoffels(self,):
        """Test equation (59)"""

        divergence_christoffels_base_point = self.stat_metric.divergence_christoffels(self.base_point)
        self.assertAllClose(divergence_christoffels_base_point, gs.zeros(divergence_christoffels_base_point.shape))
    
    def test_dual_divergence_christoffels(self,):
        pass

    def test_amari_divergence_tensor(self,):
        """Test equation (60)"""

        potential_func_first = gs.autodiff.jacobian(self.potential_function)
        potential_func_second = gs.autodiff.jacobian(potential_func_first)
        potential_func_third = gs.autodiff.jacobian(potential_func_second)
        potential_func_third_base_point = potential_func_third(self.base_point)
        amari_divergence_tensor_base_point = self.stat_metric.amari_divergence_tensor(self.base_point)
        self.assertAllClose(amari_divergence_tensor_base_point, potential_func_third_base_point)
    
