import random

from geomstats.geometry.product_positive_reals_and_poincare_disks import (
    ProductPositiveRealsAndComplexPoincareDisks,
    ProductPositiveRealsAndComplexPoincareDisksMetric,
)
from tests.data_generation import TestData, _OpenSetTestData


class ProductPositiveRealsAndComplexPoincareDisksTestData(_OpenSetTestData):

    n_manifolds_list = random.sample(range(2, 6), 2)
    dimension_list = 2 * [1]
    space_args_list = n_manifolds_list
    shape_list = list(zip(n_manifolds_list, dimension_list))
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    Space = ProductPositiveRealsAndComplexPoincareDisks

    def dimension_test_data(self):
        smoke_data = [
            dict(n_manifolds=2, expected=2),
            dict(n_manifolds=3, expected=3),
            dict(n_manifolds=4, expected=4),
        ]
        return self.generate_tests(smoke_data)


class ProductPositiveRealsAndComplexPoincareDisksMetricTestData(TestData):

    n_manifolds_list = random.sample(range(2, 6), 2)
    dimension_list = 2 * [1]
    space_args_list = n_manifolds_list
    space_list = [
        ProductPositiveRealsAndComplexPoincareDisks(*space_args)
        for space_args in space_args_list
    ]

    n_points_list = random.sample(range(1, 4), 2)
    n_tangent_vecs_list = random.sample(range(1, 4), 2)
    n_points_a_list = random.sample(range(1, 4), 2)
    n_points_b_list = [1]

    Metric = ProductPositiveRealsAndComplexPoincareDisksMetric

    def signature_test_data(self):
        smoke_data = [
            dict(n_manifolds=2, expected=(2, 0)),
            dict(n_manifolds=4, expected=(4, 0)),
        ]
        return self.generate_tests(smoke_data)

    def squared_dist_test_data(self):
        smoke_data = [dict(n_manifolds=4)]
        return self.generate_tests(smoke_data)
