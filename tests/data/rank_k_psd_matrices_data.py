import random

import geomstats.backend as gs
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.rank_k_psd_matrices import BuresWassersteinBundle, PSDMatrices
from tests.data_generation import (
    _FiberBundleTestData,
    _ManifoldTestData,
    _QuotientMetricTestData,
)


class PSDMatricesTestData(_ManifoldTestData):
    n_list = random.sample(range(3, 5), 2)
    k_list = n_list
    space_args_list = list(zip(n_list, k_list))
    shape_list = [(n, n) for n in n_list]
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    Space = PSDMatrices

    def belongs_test_data(self):
        smoke_data = [
            dict(
                n=3,
                k=2,
                mat=gs.array(
                    [
                        [0.8369314, -0.7342977, 1.0402943],
                        [0.04035992, -0.7218659, 1.0794858],
                        [0.9032698, -0.73601735, -0.36105633],
                    ]
                ),
                expected=False,
            ),
            dict(
                n=3,
                k=2,
                mat=gs.array([[1.0, 1.0, 0], [1.0, 4.0, 0], [0, 0, 0]]),
                expected=True,
            ),
        ]
        return self.generate_tests(smoke_data)


class BuresWassersteinBundleTestData(_FiberBundleTestData):
    n_list = random.sample(range(3, 5), 2)
    k_list = [n - 1 for n in n_list]

    space_args_list = list(zip(n_list, k_list))
    shape_list = [(n, n) for n in n_list]

    n_points_list = random.sample(range(1, 5), 2) * 2
    n_base_points_list = [1] * len(n_points_list) + n_points_list
    n_vecs_list = random.sample(range(1, 5), 2)

    Base = PSDMatrices
    Space = BuresWassersteinBundle


class PSDMetricBuresWassersteinTestData(_QuotientMetricTestData):
    # TODO: need to think how to solve this
    n_list = random.sample(range(3, 8), 5)

    connection_args_list = metric_args_list = [{} for _ in n_list]
    shape_list = [(n, n) for n in n_list]
    space_list = [PSDMatrices(n, n - 1) for n in n_list]

    bundle_list = [BuresWassersteinBundle(n, n - 1) for n in n_list]

    n_points_list = random.sample(range(1, 7), 5)
    n_samples_list = random.sample(range(1, 7), 5)
    n_points_a_list = random.sample(range(1, 7), 5)
    n_tangent_vecs_list = random.sample(range(1, 7), 3)
    n_points_b_list = [1]
    batch_size_list = random.sample(range(2, 7), 5)
    alpha_list = [1] * 5
    n_rungs_list = [1] * 5
    scheme_list = ["pole"] * 5

    Metric = QuotientMetric

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                bundle=BuresWassersteinBundle(3, 3),
                tangent_vec_a=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                tangent_vec_b=gs.array(
                    [[1.0, 2.0, 4.0], [2.0, 3.0, 8.0], [4.0, 8.0, 5.0]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.5, 0.5], [0.0, 0.5, 1.5]]
                ),
                expected=4.0,
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                bundle=BuresWassersteinBundle(2, 2),
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[4.0, 0.0], [0.0, 4.0]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                bundle=BuresWassersteinBundle(2, 2),
                point=gs.array([[4.0, 0.0], [0.0, 4.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[2.0, 0.0], [0.0, 2.0]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def squared_dist_test_data(self):
        smoke_data = [
            dict(
                bundle=BuresWassersteinBundle(2, 2),
                point_a=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                point_b=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                expected=2 + 4 - (2 * 2 * 2**0.5),
            )
        ]
        return self.generate_tests(smoke_data)
