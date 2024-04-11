import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad_vec
from scipy.stats import entropy

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.information_geometry.fisher_rao_metric import FisherRaoMetric
from geomstats.information_geometry.normal import NormalDistributions
from geomstats.information_geometry.statistical_metric import (
    AlphaConnection,
    DivergenceConnection,
    DualDivergenceConnection,
    StatisticalMetric,
)
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


def trapz(y, x):
    d = np.diff(x)
    return gs.sum((y[0:-1] + y[1:]) * d / 2)


euclidean_space = Euclidean(2)

normal_manifold = NormalDistributions(sample_dim=1)


def unpack_inputs(func, dim=2):
    def wrapper(tensor):
        return func(tensor[..., :dim], tensor[..., dim:])

    return wrapper


@unpack_inputs
def KL_divergence(point, base_point, eps=1e-9):
    pdf_point = normal_manifold.point_to_pdf(point)
    pdf_base_point = normal_manifold.point_to_pdf(base_point)

    pdf_point = np.vectorize(pdf_point)
    pdf_base_point = np.vectorize(pdf_base_point)

    def function_to_integrate(x):
        pdf_point_values = pdf_point(x)
        pdf_base_point_values = pdf_base_point(x)
        safe_pdf_point_values = np.maximum(pdf_point_values, eps)
        safe_pdf_base_point_values = np.maximum(pdf_base_point_values, eps)
        # Ensure the denominator is never zero
        safe_denominator = np.where(
            safe_pdf_base_point_values == 0, eps, safe_pdf_base_point_values
        )
        # return safe_pdf_point_values
        return safe_pdf_point_values * gs.log(safe_pdf_point_values)

    domain = np.linspace(-10, 10, 1000, dtype=np.float64)
    KL_div = np.trapz(function_to_integrate(domain), domain)

    return KL_div


# print(KL_divergence(gs.array([1., 2., 1., 1.])))
# domain = gs.linspace(-100, 100, 1000)
# plt.plot(domain, gs.log(normal_manifold.point_to_pdf(gs.array([0.,1.]))(domain)))
# plt.show()

# def integrate_func(thetas):
#     theta1, theta2 = thetas[:2], thetas[2:]
#     def _temp_func(x):
#         return normal_manifold.point_to_pdf(theta1)(x) * gs.log((normal_manifold.point_to_pdf(theta2)(x))/(normal_manifold.point_to_pdf(theta1)(x)))
#     domain = gs.linspace(-10, 10, 1000)
#     return trapz(_temp_func(domain), domain)


def integrate_func(theta1, theta2):
    # theta1, theta2 = thetas[:2], thetas[2:]
    def _temp_func(x):
        return normal_manifold.point_to_pdf(theta1)(x) * gs.log(
            (normal_manifold.point_to_pdf(theta1)(x))
            / (normal_manifold.point_to_pdf(theta2)(x))
        )

    domain = gs.linspace(-10, 10, 1000)
    return trapz(_temp_func(domain), domain)


stat_metric = StatisticalMetric(
    space=normal_manifold,
    divergence=integrate_func,
    primal_connection=DivergenceConnection(
        space=normal_manifold, divergence=integrate_func
    ),
    dual_connection=DualDivergenceConnection(
        space=normal_manifold, divergence=integrate_func
    ),
)

print(stat_metric.metric_matrix(gs.array([1.0, 5.0])))

# print(integrate_func(5.))
# print(gs.autodiff.hessian(integrate_func)(gs.array([1., 1., 1., 1.])))
# print(gs.autodiff.hessian(KL_divergence)(gs.array([1., 2., 1., 1.])))
fisher_metric = FisherRaoMetric(space=normal_manifold, support=[-10, 10])
print(fisher_metric.metric_matrix(gs.array([1.0, 5.0])))

# normal_stat_metric = StatisticalMetric(
#     euclidean_space,
#     divergence=KL_divergence,
#     primal_connection=DivergenceConnection(
#         space=euclidean_space, divergence=KL_divergence
#     ),
#     dual_connection=DualDivergenceConnection(
#         space=euclidean_space, divergence=KL_divergence
#     ),
# )
# base_point = gs.array([0., 1.])
# pdf_point = normal_manifold.point_to_pdf(base_point)
# domain = gs.linspace(-100, 100, 10000)
# other_pdf_point = normal_manifold.point_to_pdf(gs.array([0., 2.]))
# # return self.KL_divergence( base_point,gs.array([0.,2.])), entropy(pdf_point(domain), other_pdf_point(domain))
# # fisher_metric = self.euc_stat_metric.metric_matrix(base_point)
# KL_divergence_metric = normal_stat_metric.metric_matrix(base_point)
# print(KL_divergence_metric)
