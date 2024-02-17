"""
Test StatisticalMetric class by verifying expected induced metric, connection, and dual connection properties in the case of Bregman divergence.

Using properties from Nielson's An Elementary Introduction to Information Geometry Section 3.7 (page 14)
"""

import geomstats.backend as gs
from geomstats.information_geometry.statistical_metric import StatisticalMetric

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
stat_metric = StatisticalMetric(dim=2, divergence=breg_divergence)
base_point = gs.random.uniform(low=-10, high=10, size=(2,))

def test_metric_matrix():
    """Test equation (58) on page 15"""
    potential_hessian = gs.autodiff.hessian(potential_function)
    potential_hessian_base_point = potential_hessian(base_point)
    metric_matrix_base_point = stat_metric.metric_matrix(base_point)
    return -1*metric_matrix_base_point == potential_hessian_base_point

def test_divergence_christoffels():
    """Test equation (59) on page 15"""
    divergence_christoffels_base_point = stat_metric.divergence_christoffels(base_point)
    return divergence_christoffels_base_point == gs.zeros(divergence_christoffels_base_point.shape)


def test_amari_divergence_tensor():
    """Test equation (60) on page 15"""
    potential_func_first = gs.autodiff.jacobian(potential_function)
    potential_func_second = gs.autodiff.jacobian(potential_func_first)
    potential_func_third = gs.autodiff.jacobian(potential_func_second)
    potential_func_third_base_point = potential_func_third(base_point)
    amari_divergence_tensor_base_point = stat_metric.amari_divergence_tensor(base_point)
    return amari_divergence_tensor_base_point == potential_func_third_base_point

print(test_metric_matrix())
print(test_divergence_christoffels())
print(test_amari_divergence_tensor())