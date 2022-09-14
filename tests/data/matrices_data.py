import math
import random

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices, MatricesMetric
from tests.data_generation import _RiemannianMetricTestData, _VectorSpaceTestData

SQRT_2 = math.sqrt(2)

EYE_2 = [[1.0, 0], [0.0, 1.0]]
MINUS_EYE_2 = [[-1.0, 0], [0.0, -1.0]]
EYE_3 = [[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
MAT1_23 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
MAT2_23 = [[0.0, -2.0, -3.0], [0.0, 1.0, 1.0]]
MAT1_33 = [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]
MAT2_33 = [[1.0, 2.0, 3.0], [2.0, 4.0, 7.0], [3.0, 5.0, 6.0]]
MAT3_33 = [[0.0, 1.0, -2.0], [-1.0, 0.0, -3.0], [2.0, 3.0, 0.0]]
MAT4_33 = [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
MAT5_33 = [[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [8.0, -1.0, 2.0]]
MAT6_33 = [[1.0, 3.0, 4.0], [0.0, 2.0, 6.0], [0.0, 0.0, 2.0]]
MAT7_33 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [8.0, -1.0, 0.0]]
MAT8_33 = [[0.0, 3.0, 4.0], [0.0, 0.0, 6.0], [0.0, 0.0, 0.0]]


class MatricesTestData(_VectorSpaceTestData):
    Space = Matrices

    m_list = random.sample(range(3, 5), 2)
    n_list = random.sample(range(3, 5), 2)
    space_args_list = list(zip(m_list, n_list))
    shape_list = space_args_list
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    def belongs_test_data(self):
        sq_mat = EYE_2
        smoke_data = [
            dict(m=2, n=2, mat=sq_mat, expected=True),
            dict(m=2, n=1, mat=sq_mat, expected=False),
            dict(m=2, n=3, mat=[MAT1_23, MAT2_23], expected=[True, True]),
            dict(m=2, n=1, mat=MAT1_23, expected=False),
            dict(
                m=3,
                n=3,
                mat=[MAT1_33, MAT2_33, MAT3_33],
                expected=[True, True, True],
            ),
        ]
        return self.generate_tests(smoke_data)

    def equal_test_data(self):

        smoke_data = [
            dict(m=2, n=2, mat1=EYE_2, mat2=EYE_2, expected=True),
            dict(m=2, n=3, mat1=MAT1_23, mat2=MAT2_23, expected=False),
        ]
        return self.generate_tests(smoke_data)

    def mul_test_data(self):
        mats_1 = (
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, 2.0], [-3.0, 4.0]],
            [[1.0, -2.0], [3.0, -4.0]],
        )
        mats_2 = [[[2.0], [4.0]], [[1.0], [3.0]], [[1.0], [3.0]]]
        mat_1_x_mat_2 = [[[10.0], [22.0]], [[5.0], [9.0]], [[-5.0], [-9.0]]]
        smoke_data = [
            dict(mat=gs.array(mats_1), expected=[[23.0, -26.0], [51.0, -58.0]]),
            dict(mat=(gs.array(mats_1), gs.array(mats_2)), expected=mat_1_x_mat_2),
        ]
        return self.generate_tests(smoke_data)

    def bracket_test_data(self):
        smoke_data = [
            dict(
                mat_a=([[1.0, 2.0], [3.0, 4.0]]),
                mat_b=([[1.0, 2.0], [3.0, 4.0]]),
                expected=[[0.0, 0.0], [0.0, 0.0]],
            ),
            dict(
                mat_a=[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [0.0, 1.0]]],
                mat_b=[[[2.0, 4.0], [5.0, 4.0]], [[1.0, 4.0], [5.0, 4.0]]],
                expected=[[[-2.0, -8.0], [9.0, 2.0]], [[10.0, 6.0], [0.0, -10.0]]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def congruent_test_data(self):
        smoke_data = [
            dict(
                mat_a=[[1.0, 0.0], [2.0, -2]],
                mat_b=[[0.0, -2.0], [2.0, -3]],
                expected=[[-8.0, -20.0], [-12.0, -26.0]],
            ),
            dict(
                mat_a=[[[0.0, 1.0], [2.0, -2]], [[1.0, 0.0], [0.0, -1]]],
                mat_b=[[[1.0, -2.0], [2.0, -3]], [[0.0, 0.0], [-1.0, -3]]],
                expected=[
                    [[-14.0, -23.0], [-22.0, -36.0]],
                    [[0.0, 0.0], [0.0, -8.0]],
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def frobenius_product_test_data(self):
        smoke_data = [
            dict(
                mat_a=[[[1.0, -2.0], [1.0, 4.0]], [[1.0, 2.0], [0.0, -3.0]]],
                mat_b=[[[0.0, 4.0], [2.0, 4.0]], [[1.0, -1.0], [5.0, 4.0]]],
                expected=[10.0, -13.0],
            ),
            dict(
                mat_a=[[5.0, 8.0], [2.0, 2.0]],
                mat_b=[[0.0, 0.25], [0.5, 2.0]],
                expected=7.0,
            ),
        ]
        return self.generate_tests(smoke_data)

    def trace_product_test_data(self):
        smoke_data = [
            dict(
                mat_a=[[-2.0, 0.0], [1.0, 2.0]],
                mat_b=[[0.0, 1.0], [2.0, -2.0]],
                expected=-3.0,
            ),
            dict(
                mat_a=[[[-5.0, 0.0], [-2.0, 0.0]], [[-2.0, 1.0], [-5.0, -6.0]]],
                mat_b=[[[6.0, 5.0], [-3.0, -2.0]], [[-2.0, 0.0], [4.0, -6.0]]],
                expected=[-40.0, 44.0],
            ),
        ]
        return self.generate_tests(smoke_data)

    def flatten_test_data(self):
        smoke_data = [
            dict(m=1, n=1, mat=[[1.0]], expected=[1.0]),
            dict(m=2, n=2, mat=EYE_2, expected=[1.0, 0.0, 0.0, 1.0]),
            dict(m=2, n=3, mat=MAT1_23, expected=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            dict(
                m=2,
                n=2,
                mat=[EYE_2, MINUS_EYE_2],
                expected=[[1.0, 0.0, 0.0, 1.0], [-1.0, 0.0, 0.0, -1.0]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def flatten_reshape_test_data(self):
        random_data = [
            dict(m=1, n=1, mat=self.Space(1, 1).random_point(10000)),
            dict(m=2, n=2, mat=self.Space(2, 2).random_point(1000)),
            dict(m=2, n=10, mat=self.Space(2, 10).random_point(100)),
            dict(m=20, n=10, mat=self.Space(20, 10).random_point(100)),
        ]
        return self.generate_tests([], random_data)

    def diagonal_test_data(self):
        smoke_data = [
            dict(m=2, n=2, mat=EYE_2, expected=[1.0, 1.0]),
            dict(
                m=3,
                n=3,
                mat=[MAT1_33, MAT3_33],
                expected=[[1.0, 4.0, 6.0], [0.0, 0.0, 0.0]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def transpose_test_data(self):
        transpose_MAT3_33 = [[0.0, -1.0, 2.0], [1.0, 0.0, 3.0], [-2.0, -3.0, 0.0]]
        smoke_data = [
            dict(m=3, n=3, mat=EYE_3, expected=EYE_3),
            dict(
                m=3,
                n=3,
                mat=[MAT3_33, MAT4_33],
                expected=[transpose_MAT3_33, MAT4_33],
            ),
        ]
        return self.generate_tests(smoke_data)

    def is_diagonal_test_data(self):
        smoke_data = [
            dict(m=1, n=1, mat=[[-1.0]], expected=True),
            dict(m=2, n=2, mat=EYE_2, expected=True),
            dict(m=2, n=3, mat=MAT1_23, expected=False),
            dict(
                m=3,
                n=3,
                mat=[EYE_3, MAT3_33, MAT4_33, MAT5_33, MAT6_33, MAT7_33, MAT8_33],
                expected=[True, False, False, False, False, False, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def is_symmetric_test_data(self):
        smoke_data = [
            dict(m=1, n=1, mat=[[-1.0]], expected=True),
            dict(m=2, n=2, mat=EYE_2, expected=True),
            dict(m=2, n=3, mat=MAT1_23, expected=False),
            dict(
                m=3,
                n=3,
                mat=[MAT1_33, MAT2_33, MAT3_33],
                expected=[True, False, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def is_skew_symmetric_test_data(self):
        smoke_data = [
            dict(m=2, n=2, mat=EYE_2, expected=False),
            dict(m=2, n=3, mat=MAT1_23, expected=False),
            dict(m=2, n=2, mat=[EYE_2, MINUS_EYE_2], expected=[False, False]),
            dict(m=3, n=3, mat=[MAT2_33, MAT3_33], expected=[False, True]),
        ]
        return self.generate_tests(smoke_data)

    def is_pd_test_data(self):
        smoke_data = [
            dict(m=2, n=2, mat=EYE_2, expected=True),
            dict(m=2, n=3, mat=MAT1_23, expected=False),
            dict(m=2, n=2, mat=[EYE_2, MINUS_EYE_2], expected=[True, False]),
            dict(m=3, n=3, mat=[MAT2_33, MAT3_33], expected=[False, False]),
            dict(m=3, n=3, mat=[MAT2_33, MAT3_33], expected=[False, False]),
        ]
        return self.generate_tests(smoke_data)

    def is_spd_test_data(self):
        smoke_data = [
            dict(m=3, n=2, mat=EYE_2, expected=True),
            dict(m=3, n=3, mat=MAT4_33, expected=True),
            dict(m=2, n=3, mat=MAT1_23, expected=False),
            dict(
                m=2,
                n=2,
                mat=[EYE_2, MINUS_EYE_2],
                expected=[True, False],
            ),
            dict(
                m=3,
                n=3,
                mat=[MAT1_33, MAT2_33, MAT3_33],
                expected=[False, False, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def is_upper_triangular_test_data(self):
        smoke_data = [
            dict(m=2, n=2, mat=EYE_2, expected=True),
            dict(m=2, n=3, mat=MAT1_23, expected=False),
            dict(m=3, n=3, mat=MAT6_33, expected=True),
            dict(
                m=3,
                n=3,
                mat=[MAT1_33, MAT2_33, MAT3_33, MAT4_33, EYE_3],
                expected=[False, False, False, False, True],
            ),
        ]
        return self.generate_tests(smoke_data)

    def is_lower_triangular_test_data(self):
        smoke_data = [
            dict(m=2, n=2, mat=EYE_2, expected=True),
            dict(m=2, n=3, mat=MAT1_23, expected=False),
            dict(m=3, n=3, mat=MAT5_33, expected=True),
            dict(
                m=3,
                n=3,
                mat=[MAT1_33, MAT2_33, MAT3_33, MAT4_33, EYE_3],
                expected=[False, False, False, False, True],
            ),
        ]
        return self.generate_tests(smoke_data)

    def is_strictly_lower_triangular_test_data(self):
        smoke_data = [
            dict(m=2, n=2, mat=EYE_2, expected=False),
            dict(m=2, n=3, mat=MAT1_23, expected=False),
            dict(m=3, n=3, mat=MAT7_33, expected=True),
            dict(
                m=3,
                n=3,
                mat=[MAT1_33, MAT2_33, MAT3_33, MAT4_33, MAT5_33, MAT6_33, EYE_3],
                expected=[False, False, False, False, False, False, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def is_strictly_upper_triangular_test_data(self):
        smoke_data = [
            dict(m=2, n=2, mat=EYE_2, expected=False),
            dict(m=2, n=3, mat=MAT1_23, expected=False),
            dict(m=3, n=3, mat=MAT8_33, expected=True),
            dict(
                m=3,
                n=3,
                mat=[MAT1_33, MAT2_33, MAT3_33, MAT4_33, MAT5_33, MAT6_33, EYE_3],
                expected=[False, False, False, False, False, False, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_diagonal_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                mat=[[1.0, 2.0], [3.0, 4.0]],
                expected=[[1.0, 0.0], [0.0, 4.0]],
            ),
            dict(
                m=2,
                n=2,
                mat=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                expected=[[[1.0, 0.0], [0.0, 4.0]], [[5.0, 0.0], [0.0, 8.0]]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_symmetric_test_data(self):
        res = 0.5 * (1e100 + 1e-100)
        smoke_data = [
            dict(
                m=2,
                n=2,
                mat=[[1.0, 2.0], [2.0, 1.0]],
                expected=[[1.0, 2.0], [2.0, 1.0]],
            ),
            dict(
                m=3,
                n=3,
                mat=[
                    [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [3.0, 1.0, 1.0]],
                    [
                        [1e100, 1e-100, 1e100],
                        [1e100, 1e-100, 1e100],
                        [1e-100, 1e-100, 1e100],
                    ],
                ],
                expected=[
                    [[1.0, 1.0, 3.0], [1.0, 0.0, 0.5], [3.0, 0.5, 1.0]],
                    [[1e100, res, res], [res, 1e-100, res], [res, res, 1e100]],
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_lower_triangular_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                mat=[[1.0, 2.0], [3.0, 4.0]],
                expected=[[1.0, 0.0], [3.0, 4.0]],
            ),
            dict(
                m=2,
                n=2,
                mat=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                expected=[[[1.0, 0.0], [3.0, 4.0]], [[5.0, 0.0], [7.0, 8.0]]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_upper_triangular_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                mat=[[1.0, 2.0], [3.0, 4.0]],
                expected=[[1.0, 2.0], [0.0, 4.0]],
            ),
            dict(
                m=2,
                n=2,
                mat=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                expected=[[[1.0, 2.0], [0.0, 4.0]], [[5.0, 6.0], [0.0, 8.0]]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_strictly_lower_triangular_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                mat=[[1.0, 2.0], [3.0, 4.0]],
                expected=[[0.0, 0.0], [3.0, 0.0]],
            ),
            dict(
                m=2,
                n=2,
                mat=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                expected=[[[0.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [7.0, 0.0]]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_strictly_upper_triangular_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                mat=[[1.0, 2.0], [3.0, 4.0]],
                expected=[[0.0, 2.0], [0.0, 0.0]],
            ),
            dict(
                m=2,
                n=2,
                mat=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                expected=[[[0.0, 2.0], [0.0, 0.0]], [[0.0, 6.0], [0.0, 0.0]]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_lower_triangular_diagonal_scaled_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                mat=[[2.0, 2.0], [3.0, 4.0]],
                expected=[[1.0, 0.0], [3.0, 2.0]],
            ),
            dict(
                m=2,
                n=2,
                mat=[[[2.0, 2.0], [3.0, 4.0]], [[6.0, 6.0], [7.0, 8.0]]],
                expected=[[[1.0, 0], [3.0, 2.0]], [[3.0, 0.0], [7.0, 4.0]]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_matrix_type_is_matrix_type_test_data(self):
        matrix_types = [
            "diagonal",
            "symmetric",
            "skew_symmetric",
            "lower_triangular",
            "upper_triangular",
            "strictly_lower_triangular",
            "strictly_upper_triangular",
        ]
        list_n = random.sample(range(1, 100), 50)
        n_samples = 50
        random_data = []
        for matrix_type in matrix_types:
            for n in list_n:
                mat = gs.random.normal(size=(n_samples, n, n))
                random_data += [dict(m=n, n=n, matrix_type=matrix_type, mat=mat)]
        return self.generate_tests([], random_data)

    def basis_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                expected=gs.array(
                    [
                        gs.array_from_sparse([(i, j)], [1], (2, 2))
                        for i in range(2)
                        for j in range(2)
                    ]
                ),
            ),
            dict(
                m=2,
                n=3,
                expected=gs.array(
                    [
                        gs.array_from_sparse([(i, j)], [1], (2, 3))
                        for i in range(2)
                        for j in range(3)
                    ]
                ),
            ),
        ]
        return self.generate_tests(smoke_data)


class MatricesMetricTestData(_RiemannianMetricTestData):
    m_list = random.sample(range(3, 5), 2)
    n_list = random.sample(range(3, 5), 2)
    metric_args_list = list(zip(m_list, n_list))
    space_args_list = metric_args_list
    shape_list = space_args_list
    space_list = [Matrices(m, n) for m, n in metric_args_list]
    n_points_list = random.sample(range(1, 7), 2)
    n_tangent_vecs_list = random.sample(range(1, 7), 2)
    n_points_a_list = random.sample(range(1, 7), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = MatricesMetric

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                tangent_vec_a=[[-3.0, 1.0], [-1.0, -2.0]],
                tangent_vec_b=[[-9.0, 0.0], [4.0, 2.0]],
                expected=19.0,
            ),
            dict(
                m=2,
                n=2,
                tangent_vec_a=[
                    [[-1.5, 0.0], [2.0, -3.0]],
                    [[0.5, 7.0], [0.5, -2.0]],
                ],
                tangent_vec_b=[
                    [[2.0, 0.0], [2.0, -3.0]],
                    [[-1.0, 0.0], [1.0, -2.0]],
                ],
                expected=[10.0, 4.0],
            ),
        ]
        return self.generate_tests(smoke_data)

    def norm_test_data(self):
        smoke_data = [
            dict(m=2, n=2, vector=[[1.0, 0.0], [0.0, 1.0]], expected=SQRT_2),
            dict(
                m=2,
                n=2,
                vector=[[[3.0, 0.0], [4.0, 0.0]], [[-3.0, 0.0], [-4.0, 0.0]]],
                expected=[5.0, 5.0],
            ),
        ]
        return self.generate_tests(smoke_data)

    def inner_product_norm_test_data(self):
        smoke_data = [
            dict(m=5, n=5, mat=Matrices(5, 5).random_point(100)),
            dict(m=10, n=10, mat=Matrices(5, 5).random_point(100)),
        ]
        return self.generate_tests(smoke_data)
