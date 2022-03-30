"""Unit tests for the manifold of matrices."""
import math
import random

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.matrices import Matrices, MatricesMetric
from tests.conftest import Parametrizer
from tests.data_generation import _RiemannianMetricTestData, _VectorSpaceTestData
from tests.geometry_test_cases import RiemannianMetricTestCase, VectorSpaceTestCase

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


class TestMatrices(VectorSpaceTestCase, metaclass=Parametrizer):
    space = Matrices

    class MatricesTestData(_VectorSpaceTestData):
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
                dict(m=2, n=2, mat_1=EYE_2, mat_2=EYE_2, expected=True),
                dict(m=2, n=3, mat_1=MAT1_23, mat_2=MAT2_23, expected=False),
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
                    mat_1=[[1.0, 0.0], [2.0, -2]],
                    mat_2=[[0.0, -2.0], [2.0, -3]],
                    expected=[[-8.0, -20.0], [-12.0, -26.0]],
                ),
                dict(
                    mat_1=[[[0.0, 1.0], [2.0, -2]], [[1.0, 0.0], [0.0, -1]]],
                    mat_2=[[[1.0, -2.0], [2.0, -3]], [[0.0, 0.0], [-1.0, -3]]],
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
                dict(m=1, n=1, mat=Matrices(1, 1).random_point(10000)),
                dict(m=2, n=2, mat=Matrices(2, 2).random_point(1000)),
                dict(m=2, n=10, mat=Matrices(2, 10).random_point(100)),
                dict(m=20, n=10, mat=Matrices(20, 10).random_point(100)),
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

        def basis_belongs_test_data(self):
            return self._basis_belongs_test_data(self.space_args_list)

        def basis_cardinality_test_data(self):
            return self._basis_cardinality_test_data(self.space_args_list)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(2, 2), (3, 2)]
            smoke_n_points_list = [1, 2]
            belongs_atol = gs.atol * 10000
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
                belongs_atol,
            )

        def projection_belongs_test_data(self):
            belongs_atol = gs.atol * 1000
            return self._projection_belongs_test_data(
                self.space_args_list, self.shape_list, self.n_points_list, belongs_atol
            )

        def to_tangent_is_tangent_test_data(self):
            is_tangent_atol = gs.atol * 1000
            return self._to_tangent_is_tangent_test_data(
                Matrices,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
                is_tangent_atol,
            )

        def basis_test_data(self):
            smoke_data = [
                dict(
                    n=2,
                    m=2,
                    expected=gs.array(
                        [
                            gs.array_from_sparse([(i, j)], [1], (2, 2))
                            for i in range(2)
                            for j in range(2)
                        ]
                    ),
                ),
                dict(
                    n=2,
                    m=3,
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

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                Matrices, self.space_args_list, self.n_vecs_list
            )

    testing_data = MatricesTestData()

    def test_belongs(self, m, n, mat, expected):
        self.assertAllClose(Matrices(m, n).belongs(gs.array(mat)), gs.array(expected))

    def test_equal(self, m, n, mat1, mat2, expected):
        self.assertAllClose(
            Matrices(m, n).equal(gs.array(mat1), gs.array(mat2)), gs.array(expected)
        )

    def test_mul(self, mat, expected):
        self.assertAllClose(Matrices.mul(*mat), gs.array(expected))

    def test_bracket(self, mat_a, mat_b, expected):
        self.assertAllClose(
            Matrices.bracket(gs.array(mat_a), gs.array(mat_b)), gs.array(expected)
        )

    def test_congruent(self, mat_a, mat_b, expected):
        self.assertAllClose(
            Matrices.congruent(gs.array(mat_a), gs.array(mat_b)), gs.array(expected)
        )

    def test_frobenius_product(self, mat_a, mat_b, expected):
        self.assertAllClose(
            Matrices.frobenius_product(gs.array(mat_a), gs.array(mat_b)),
            gs.array(expected),
        )

    def test_trace_product(self, mat_a, mat_b, expected):
        self.assertAllClose(
            Matrices.trace_product(gs.array(mat_a), gs.array(mat_b)), gs.array(expected)
        )

    def test_flatten(self, m, n, mat, expected):
        self.assertAllClose(Matrices(m, n).flatten(gs.array(mat)), gs.array(expected))

    def test_transpose(self, m, n, mat, expected):
        self.assertAllClose(Matrices(m, n).transpose(gs.array(mat)), gs.array(expected))

    def test_diagonal(self, m, n, mat, expected):
        self.assertAllClose(Matrices(m, n).diagonal(gs.array(mat)), gs.array(expected))

    def test_is_diagonal(self, m, n, mat, expected):
        self.assertAllClose(Matrices(m, n).is_diagonal(gs.array(mat)), expected)

    def test_is_symmetric(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).is_symmetric(gs.array(mat)), gs.array(expected)
        )

    def test_is_skew_symmetric(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).is_skew_symmetric(gs.array(mat)), gs.array(expected)
        )

    def test_is_pd(self, m, n, mat, expected):
        self.assertAllClose(Matrices(m, n).is_pd(gs.array(mat)), gs.array(expected))

    def test_is_spd(self, m, n, mat, expected):
        self.assertAllClose(Matrices(m, n).is_spd(gs.array(mat)), gs.array(expected))

    def test_is_upper_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).is_upper_triangular(gs.array(mat)), gs.array(expected)
        )

    def test_is_lower_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).is_lower_triangular(gs.array(mat)), gs.array(expected)
        )

    def test_is_strictly_lower_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).is_strictly_lower_triangular(gs.array(mat)),
            gs.array(expected),
        )

    def test_is_strictly_upper_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).is_strictly_upper_triangular(gs.array(mat)),
            gs.array(expected),
        )

    def test_to_diagonal(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).to_diagonal(gs.array(mat)), gs.array(expected)
        )

    @geomstats.tests.np_autograd_and_torch_only
    def test_to_symmetric(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).to_symmetric(gs.array(mat)), gs.array(expected)
        )

    def test_to_lower_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).to_lower_triangular(gs.array(mat)), gs.array(expected)
        )

    def test_to_upper_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).to_upper_triangular(gs.array(mat)), gs.array(expected)
        )

    def test_to_strictly_lower_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).to_strictly_lower_triangular(gs.array(mat)),
            gs.array(expected),
        )

    def test_to_strictly_upper_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).to_strictly_upper_triangular(gs.array(mat)),
            gs.array(expected),
        )

    def test_to_lower_triangular_diagonal_scaled(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).to_lower_triangular_diagonal_scaled(gs.array(mat)),
            gs.array(expected),
        )

    def test_flatten_reshape(self, m, n, mat):
        cls_mn = Matrices(m, n)
        self.assertAllClose(
            cls_mn.reshape(cls_mn.flatten(gs.array(mat))), gs.array(mat)
        )

    def test_to_matrix_type_is_matrix_type(self, m, n, matrix_type, mat):
        cls_mn = Matrices(m, n)
        to_function = getattr(cls_mn, "to_" + matrix_type)
        is_function = getattr(cls_mn, "is_" + matrix_type)
        self.assertAllClose(gs.all(is_function(to_function(gs.array(mat)))), True)

    def test_basis(self, m, n, expected):
        result = Matrices(m, n).basis
        self.assertAllClose(result, expected)


class TestMatricesMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = MatricesMetric
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_exp_geodesic_ivp = True

    class MatricesMetricTestData(_RiemannianMetricTestData):
        m_list = random.sample(range(3, 5), 2)
        n_list = random.sample(range(3, 5), 2)
        metric_args_list = list(zip(m_list, n_list))
        space_args_list = metric_args_list
        shape_list = space_args_list
        space_list = [Matrices(m, n) for m, n in metric_args_list]
        n_points_list = random.sample(range(1, 7), 5)
        n_tangent_vecs_list = random.sample(range(1, 7), 5)
        n_points_a_list = random.sample(range(1, 7), 5)
        n_points_b_list = [1]
        alpha_list = [1] * 5
        n_rungs_list = [1] * 5
        scheme_list = ["pole"] * 5

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

        def exp_shape_test_data(self):
            return self._exp_shape_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
            )

        def log_shape_test_data(self):
            return self._log_shape_test_data(
                self.metric_args_list,
                self.space_list,
            )

        def squared_dist_is_symmetric_test_data(self):
            return self._squared_dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
                atol=gs.atol * 1000,
            )

        def exp_belongs_test_data(self):
            return self._exp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_is_tangent_test_data(self):
            return self._log_is_tangent_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                is_tangent_atol=gs.atol * 1000,
            )

        def geodesic_ivp_belongs_test_data(self):
            return self._geodesic_ivp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def geodesic_bvp_belongs_test_data(self):
            return self._geodesic_bvp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_then_exp_test_data(self):
            return self._log_then_exp_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def exp_then_log_test_data(self):
            return self._exp_then_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def exp_ladder_parallel_transport_test_data(self):
            return self._exp_ladder_parallel_transport_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_rungs_list,
                self.alpha_list,
                self.scheme_list,
            )

        def exp_geodesic_ivp_test_data(self):
            return self._exp_geodesic_ivp_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_points_list,
                rtol=gs.rtol * 100000,
                atol=gs.atol * 100000,
            )

        def parallel_transport_ivp_is_isometry_test_data(self):
            return self._parallel_transport_ivp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def parallel_transport_bvp_is_isometry_test_data(self):
            return self._parallel_transport_bvp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

    testing_data = MatricesMetricTestData()

    def test_inner_product(self, m, n, tangent_vec_a, tangent_vec_b, expected):
        self.assertAllClose(
            self.metric(m, n).inner_product(
                gs.array(tangent_vec_a), gs.array(tangent_vec_b)
            ),
            gs.array(expected),
        )

    def test_norm(self, m, n, vector, expected):
        self.assertAllClose(
            self.metric(m, n).norm(gs.array(vector)), gs.array(expected)
        )

    def test_inner_product_norm(self, m, n, mat):
        self.assertAllClose(
            self.metric(m, n).inner_product(mat, mat),
            gs.power(self.metric(m, n).norm(mat), 2),
        )
