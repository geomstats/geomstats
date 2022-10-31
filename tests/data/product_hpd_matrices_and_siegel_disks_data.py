import random

import geomstats.backend as gs
from geomstats.geometry.product_hpd_matrices_and_siegel_disks import (
    ProductHPDMatricesAndSiegelDisks,
    ProductHPDMatricesAndSiegelDisksMetric,
)
from tests.data_generation import TestData, _OpenSetTestData


class ProductHPDMatricesAndSiegelDisksTestData(_OpenSetTestData):

    n_manifolds_list = random.sample(range(2, 6), 2)
    dimension_list = random.sample(range(2, 6), 2)
    # space_parameters = list(zip(n_manifolds_list, dimension_list))
    # space_args_list = [(n_manifolds,) for n_manifolds in n_manifolds_list]
    space_args_list = list(zip(n_manifolds_list, dimension_list))
    shape_list = [(n_manifolds, 3) for n_manifolds in n_manifolds_list]
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    Space = ProductHPDMatricesAndSiegelDisks

    def dimension_test_data(self):
        smoke_data = [
            dict(n_manifolds=2, n=2, expected=8),
            dict(n_manifolds=3, n=5, expected=75),
            dict(n_manifolds=4, n=7, expected=196),
        ]
        return self.generate_tests(smoke_data)


class ProductHPDMatricesAndSiegelDisksMetricTestData(TestData):

    # n_manifolds_list = random.sample(range(2, 4), 2)
    # metric_args_list = [(n_manifolds,) for n_manifolds in n_manifolds_list]
    # shape_list = [(n_manifolds, 3) for n_manifolds in n_manifolds_list]
    # space_list = [ProductHPDMatricesAndSiegelDisks(n_manifolds, n) for n_manifolds in n_manifolds_list]
    n_manifolds_list = random.sample(range(2, 6), 2)
    dimension_list = random.sample(range(2, 6), 2)
    space_args_list = list(zip(n_manifolds_list, dimension_list))
    # metric_args_list = [(n_manifolds,) for n_manifolds in n_manifolds_list]
    # shape_list = [(n_manifolds, 3) for n_manifolds in n_manifolds_list]
    space_list = [
        ProductHPDMatricesAndSiegelDisks(*space_args) for space_args in space_args_list
    ]

    n_points_list = random.sample(range(1, 4), 2)
    n_tangent_vecs_list = random.sample(range(1, 4), 2)
    n_points_a_list = random.sample(range(1, 4), 2)
    n_points_b_list = [1]
    # alpha_list = [1] * 2
    # n_rungs_list = [1] * 2
    # scheme_list = ["pole"] * 2

    Metric = ProductHPDMatricesAndSiegelDisksMetric

    def signature_test_data(self):
        smoke_data = [
            dict(n_manifolds=2, n=5, expected=(50, 0)),
            dict(n_manifolds=4, n=3, expected=(36, 0)),
        ]
        return self.generate_tests(smoke_data)
