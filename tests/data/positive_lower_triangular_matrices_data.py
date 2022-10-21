import math
import random

import geomstats.backend as gs
from geomstats.geometry.positive_lower_triangular_matrices import (
    CholeskyMetric,
    PositiveLowerTriangularMatrices,
)
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData

EULER = gs.exp(1.0)
SQRT_2 = math.sqrt(2)


class PositiveLowerTriangularMatricesTestData(_OpenSetTestData):
    n_list = random.sample(range(2, 5), 2)
    space_args_list = [(n,) for n in n_list]
    shape_list = [(n, n) for n in n_list]
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)
    batch_shape_list = [
        tuple(random.choices(range(2, 10), k=i)) for i in random.sample(range(1, 5), 3)
    ]

    Space = PositiveLowerTriangularMatrices

    def belongs_test_data(self):
        smoke_data = [
            dict(n=2, mat=[[1.0, 0.0], [-1.0, 3.0]], expected=True),
            dict(n=2, mat=[[1.0, -1.0], [-1.0, 3.0]], expected=False),
            dict(n=2, mat=[[-1.0, 0.0], [-1.0, 3.0]], expected=False),
            dict(n=3, mat=[[1.0, 0], [0, 1.0]], expected=False),
            dict(
                n=2,
                mat=[
                    [[1.0, 0], [0, 1.0]],
                    [[1.0, 2.0], [2.0, 1.0]],
                    [[-1.0, 0.0], [1.0, 1.0]],
                    [[0.0, 0.0], [1.0, 1.0]],
                ],
                expected=[True, False, False, False],
            ),
            dict(
                n=3,
                mat=[
                    [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[-1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ],
                expected=[False, False, True, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def random_point_and_belongs_test_data(self):
        smoke_data = [
            dict(n=1, n_samples=1),
            dict(n=2, n_samples=2),
            dict(n=10, n_samples=100),
            dict(n=100, n_samples=10),
        ]
        return self.generate_tests(smoke_data)

    def gram_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point=[[1.0, 0.0], [2.0, 1.0]],
                expected=[[1.0, 2.0], [2.0, 5.0]],
            ),
            dict(
                n=2,
                point=[[[2.0, 1.0], [0.0, 1.0]], [[-6.0, 0.0], [5.0, 3.0]]],
                expected=[[[5.0, 1.0], [1.0, 1.0]], [[36.0, -30.0], [-30.0, 34.0]]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def differential_gram_test_data(self):
        smoke_data = [
            dict(
                n=2,
                tangent_vec=[[-1.0, 0.0], [2.0, -1.0]],
                base_point=[[1.0, 0.0], [2.0, 1.0]],
                expected=[[-2.0, 0.0], [0.0, 6.0]],
            ),
            dict(
                n=2,
                tangent_vec=[[[-1.0, 2.0], [2.0, -1.0]], [[0.0, 4.0], [4.0, -1.0]]],
                base_point=[[[3.0, 0.0], [-1.0, 2.0]], [[4.0, 0.0], [-1.0, 4.0]]],
                expected=[
                    [[-6.0, 11.0], [11.0, -8.0]],
                    [[0.0, 32.0], [32.0, -16.0]],
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def inverse_differential_gram_test_data(self):
        smoke_data = [
            dict(
                n=2,
                tangent_vec=[[1.0, 2.0], [2.0, 5.0]],
                base_point=[[1.0, 0.0], [2.0, 2.0]],
                expected=[[0.5, 0.0], [1.0, 0.25]],
            ),
            dict(
                n=2,
                tangent_vec=[[[-4.0, 1.0], [1.0, -4.0]], [[0.0, 4.0], [4.0, -8.0]]],
                base_point=[[[2.0, 0.0], [-1.0, 2.0]], [[4.0, 0.0], [-1.0, 2.0]]],
                expected=[[[-1.0, 0.0], [0.0, -1.0]], [[0.0, 0.0], [1.0, -1.5]]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def differential_gram_belongs_test_data(self):
        n_list = [1, 2, 2, 3, 10]
        n_samples_list = [1, 1, 2, 10, 5]
        random_data = [
            dict(
                n=n,
                tangent_vec=self.Space(n).embedding_space.random_point(n_samples),
                base_point=self.Space(n).random_point(n_samples),
            )
            for n, n_samples in zip(n_list, n_samples_list)
        ]
        return self.generate_tests([], random_data)

    def inverse_differential_gram_belongs_test_data(self):
        n_list = [1, 2, 2, 3, 10]
        n_samples_list = [1, 1, 2, 10, 5]
        random_data = [
            dict(
                n=n,
                tangent_vec=self.Space(n).embedding_space.random_point(n_samples),
                base_point=self.Space(n).random_point(n_samples),
            )
            for n, n_samples in zip(n_list, n_samples_list)
        ]
        return self.generate_tests([], random_data)


class CholeskyMetricTestData(_RiemannianMetricTestData):
    n_list = random.sample(range(2, 5), 2)
    metric_args_list = [(n,) for n in n_list]
    shape_list = [(n, n) for n in n_list]
    space_list = [PositiveLowerTriangularMatrices(n) for n in n_list]
    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    batch_size_list = random.sample(range(2, 5), 2)
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = CholeskyMetric

    def diag_inner_product_test_data(self):
        smoke_data = [
            dict(
                n=2,
                tangent_vec_a=[[1.0, 0.0], [-2.0, -1.0]],
                tangent_vec_b=[[2.0, 0.0], [-3.0, -1.0]],
                base_point=[[SQRT_2, 0.0], [-3.0, 1.0]],
                expected=2.0,
            )
        ]
        return self.generate_tests(smoke_data)

    def strictly_lower_inner_product_test_data(self):
        smoke_data = [
            dict(
                n=2,
                tangent_vec_a=[[1.0, 0.0], [-2.0, -1.0]],
                tangent_vec_b=[[2.0, 0.0], [-3.0, -1.0]],
                expected=6.0,
            )
        ]
        return self.generate_tests(smoke_data)

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                n=2,
                tangent_vec_a=[[1.0, 0.0], [-2.0, -1.0]],
                tangent_vec_b=[[2.0, 0.0], [-3.0, -1.0]],
                base_point=[[SQRT_2, 0.0], [-3.0, 1.0]],
                expected=8.0,
            ),
            dict(
                n=2,
                tangent_vec_a=[
                    [[3.0, 0.0], [4.0, 2.0]],
                    [[-1.0, 0.0], [2.0, -4.0]],
                ],
                tangent_vec_b=[[[4.0, 0.0], [3.0, 3.0]], [[3.0, 0.0], [-6.0, 2.0]]],
                base_point=[[[3, 0.0], [-2.0, 6.0]], [[1, 0.0], [-1.0, 1.0]]],
                expected=[13.5, -23.0],
            ),
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                n=2,
                tangent_vec=[[-1.0, 0.0], [2.0, 3.0]],
                base_point=[[1.0, 0.0], [2.0, 2.0]],
                expected=[[1 / EULER, 0.0], [4.0, 2 * gs.exp(1.5)]],
            ),
            dict(
                n=2,
                tangent_vec=[[[0.0, 0.0], [2.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]],
                base_point=[[[1.0, 0.0], [2.0, 2.0]], [[1.0, 0.0], [0.0, 2.0]]],
                expected=[
                    [[1.0, 0.0], [4.0, 2.0]],
                    [[gs.exp(1.0), 0.0], [0.0, 2.0]],
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point=[[EULER, 0.0], [2.0, EULER**3]],
                base_point=[[EULER**3, 0.0], [4.0, EULER**4]],
                expected=[[-2.0 * EULER**3, 0.0], [-2.0, -1 * EULER**4]],
            ),
            dict(
                n=2,
                point=[
                    [[gs.exp(-2.0), 0.0], [0.0, gs.exp(2.0)]],
                    [[gs.exp(-3.0), 0.0], [2.0, gs.exp(3.0)]],
                ],
                base_point=[[[1.0, 0.0], [-1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
                expected=[[[-2.0, 0.0], [1.0, 2.0]], [[-3.0, 0.0], [2.0, 3.0]]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def squared_dist_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point_a=[[EULER, 0.0], [2.0, EULER**3]],
                point_b=[[EULER**3, 0.0], [4.0, EULER**4]],
                expected=9,
            ),
            dict(
                n=2,
                point_a=[
                    [[EULER, 0.0], [2.0, EULER**3]],
                    [[EULER, 0.0], [4.0, EULER**3]],
                ],
                point_b=[
                    [[EULER**3, 0.0], [4.0, EULER**4]],
                    [[EULER**3, 0.0], [7.0, EULER**4]],
                ],
                expected=[9, 14],
            ),
        ]
        return self.generate_tests(smoke_data)
