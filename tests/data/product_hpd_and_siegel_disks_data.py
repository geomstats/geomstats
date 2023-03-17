import random

from geomstats.geometry.product_hpd_and_siegel_disks import (
    ProductHPDMatricesAndSiegelDisks,
    ProductHPDMatricesAndSiegelDisksMetric,
)
from tests.data_generation import TestData, _OpenSetTestData


class ProductHPDMatricesAndSiegelDisksTestData(_OpenSetTestData):

    n_manifolds_list = random.sample(range(2, 6), 2)
    dimension_list = random.sample(range(2, 6), 2)
    space_args_list = list(zip(n_manifolds_list, dimension_list))
    shape_list = list(zip(n_manifolds_list, dimension_list, dimension_list))
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

    n_manifolds_list = random.sample(range(2, 6), 2)
    dimension_list = random.sample(range(2, 6), 2)
    space_args_list = list(zip(n_manifolds_list, dimension_list))
    space_list = [
        ProductHPDMatricesAndSiegelDisks(*space_args, equip=False)
        for space_args in space_args_list
    ]

    n_points_list = random.sample(range(1, 4), 2)
    n_tangent_vecs_list = random.sample(range(1, 4), 2)
    n_points_a_list = random.sample(range(1, 4), 2)
    n_points_b_list = [1]

    Metric = ProductHPDMatricesAndSiegelDisksMetric

    def signature_test_data(self):
        smoke_data = [
            dict(
                space=ProductHPDMatricesAndSiegelDisks(2, 5, equip=False),
                expected=(50, 0),
            ),
            dict(
                space=ProductHPDMatricesAndSiegelDisks(4, 3, equip=False),
                expected=(36, 0),
            ),
        ]
        return self.generate_tests(smoke_data)

    def squared_dist_test_data(self):
        smoke_data = [dict(space=ProductHPDMatricesAndSiegelDisks(4, 2, equip=False))]
        return self.generate_tests(smoke_data)
