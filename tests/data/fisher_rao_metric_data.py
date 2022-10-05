"""Test data for the fisher rao metric."""

import geomstats.backend as gs
from geomstats.information_geometry.fisher_rao_metric import FisherRaoMetric
from geomstats.information_geometry.normal import NormalDistributions
from tests.data_generation import _RiemannianMetricTestData


class FisherRaoMetricTestData(_RiemannianMetricTestData):
    information_manifolds = [
        NormalDistributions(),
    ]
    supports = [(-10, 10)]
    Metric = FisherRaoMetric
    metric_args_list = [
        (information_manifold, support)
        for information_manifold, support in zip(information_manifolds, supports)
    ]
    shape_list = [metric_args[0].shape for metric_args in metric_args_list]
    space_list = [metric_args[0] for metric_args in metric_args_list]
    n_points_list = [1, 2] * 3
    n_tangent_vecs_list = [1, 2] * 3
    n_points_a_list = [1, 2] * 3
    n_points_b_list = [1]
    alpha_list = [1] * 6
    n_rungs_list = [1] * 6
    scheme_list = ["pole"] * 6

    def inner_product_matrix_shape_test_data(self):
        smoke_data = [
            dict(
                information_manifold=NormalDistributions(),
                support=(-10, 10),
                base_point=gs.array([1.0, 2.0]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def inner_product_matrix_and_its_inverse_test_data(self):
        smoke_data = [
            dict(
                information_manifold=NormalDistributions(),
                support=(-10, 10),
                base_point=gs.array([1.0, 2.0]),
            ),
        ]
        return self.generate_tests(smoke_data)
