import math
import random

import geomstats.backend as gs
from geomstats.geometry.complex_matrices import ComplexMatrices
from tests.data_generation import _ComplexVectorSpaceTestData, _RiemannianMetricTestData

SQRT_2 = math.sqrt(2)

MAT1_11 = gs.array([[-1.0]], dtype=gs.complex128)
EYE_2 = gs.array([[1.0, 0], [0.0, 1.0]], dtype=gs.complex128)
MINUS_EYE_2 = gs.array([[-1.0, 0], [0.0, -1.0]], dtype=gs.complex128)
MAT1_22 = gs.array([[1.0, 2.0], [2.0, 1.0]], dtype=gs.complex128)
EYE_3 = gs.array([[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=gs.complex128)
MAT1_23 = gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=gs.complex128)
MAT2_23 = gs.array([[0.0, -2.0, -3.0], [0.0, 1.0, 1.0]], dtype=gs.complex128)
MAT1_33 = gs.array(
    [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]], dtype=gs.complex128
)
MAT2_33 = gs.array(
    [[1.0, 2.0, 3.0], [2.0, 4.0, 7.0], [3.0, 5.0, 6.0]], dtype=gs.complex128
)
MAT3_33 = gs.array(
    [[0.0, 1.0, -2.0], [-1.0, 0.0, -3.0], [2.0, 3.0, 0.0]], dtype=gs.complex128
)
MAT4_33 = gs.array(
    [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]], dtype=gs.complex128
)
MAT5_33 = gs.array(
    [[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [8.0, -1.0, 2.0]], dtype=gs.complex128
)
MAT6_33 = gs.array(
    [[1.0, 3.0, 4.0], [0.0, 2.0, 6.0], [0.0, 0.0, 2.0]], dtype=gs.complex128
)
MAT7_33 = gs.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [8.0, -1.0, 0.0]], dtype=gs.complex128
)
MAT8_33 = gs.array(
    [[0.0, 3.0, 4.0], [0.0, 0.0, 6.0], [0.0, 0.0, 0.0]], dtype=gs.complex128
)
MAT9_33 = gs.array([[4, 0, 1j], [0, 3, 0], [-1j, 0, 4]], dtype=gs.complex128)
MAT10_33 = gs.array(
    [[1, 1 + 1j, -3j], [1 - 1j, 0, 4], [3j, 4, -2]], dtype=gs.complex128
)
MAT11_33 = gs.array([[1, 1j, 2], [1j, -1j, 2], [4 + 2j, 0, -2]], dtype=gs.complex128)
MAT12_33 = gs.array([[1, -1j, 4 - 2j], [-1j, 1j, 0], [2, 2, -2]], dtype=gs.complex128)
MAT13_33 = gs.array([[1, 0, 3 - 1j], [0, 0, 1], [3 + 1j, 1, -2]], dtype=gs.complex128)


class ComplexMatricesTestData(_ComplexVectorSpaceTestData):
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
        mats_1 = gs.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[-1.0, 2.0], [-3.0, 4.0]],
                [[1.0, -2.0], [3.0, -4.0]],
            ],
            dtype=gs.complex128,
        )
        mats_2 = gs.array(
            [[[2.0], [4.0]], [[1.0], [3.0]], [[1.0], [3.0]]], dtype=gs.complex128
        )
        mat_1_x_mat_2 = gs.array(
            [[[10.0], [22.0]], [[5.0], [9.0]], [[-5.0], [-9.0]]], dtype=gs.complex128
        )
        smoke_data = [
            dict(
                mat=mats_1,
                expected=gs.array([[23.0, -26.0], [51.0, -58.0]], dtype=gs.complex128),
            ),
            dict(mat=(mats_1, mats_2), expected=mat_1_x_mat_2),
        ]
        return self.generate_tests(smoke_data)

    def bracket_test_data(self):
        smoke_data = [
            dict(
                mat_a=gs.array([[1.0, 2.0], [3.0, 4.0]], dtype=gs.complex128),
                mat_b=gs.array([[1.0, 2.0], [3.0, 4.0]], dtype=gs.complex128),
                expected=gs.array([[0.0, 0.0], [0.0, 0.0]], dtype=gs.complex128),
            ),
            dict(
                mat_a=[
                    gs.array([[1.0, 2.0], [3.0, 4.0]], dtype=gs.complex128),
                    gs.array([[1.0, 2.0], [0.0, 1.0]], dtype=gs.complex128),
                ],
                mat_b=[
                    gs.array([[2.0, 4.0], [5.0, 4.0]], dtype=gs.complex128),
                    gs.array([[1.0, 4.0], [5.0, 4.0]], dtype=gs.complex128),
                ],
                expected=[
                    gs.array([[-2.0, -8.0], [9.0, 2.0]], dtype=gs.complex128),
                    gs.array([[10.0, 6.0], [0.0, -10.0]], dtype=gs.complex128),
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def congruent_test_data(self):
        smoke_data = [
            dict(
                mat_1=gs.array([[1.0, 0.0], [2.0, -2]], dtype=gs.complex128),
                mat_2=gs.array([[0.0, -2.0], [2.0, -3]], dtype=gs.complex128),
                expected=gs.array([[-8.0, -20.0], [-12.0, -26.0]], dtype=gs.complex128),
            ),
            dict(
                mat_1=[
                    gs.array([[0.0, 1.0], [2.0, -2]], dtype=gs.complex128),
                    gs.array([[1.0, 0.0], [0.0, -1]], dtype=gs.complex128),
                ],
                mat_2=[
                    gs.array([[1.0, -2.0], [2.0, -3]], dtype=gs.complex128),
                    gs.array([[0.0, 0.0], [-1.0, -3]], dtype=gs.complex128),
                ],
                expected=[
                    gs.array([[-14.0, -23.0], [-22.0, -36.0]], dtype=gs.complex128),
                    gs.array([[0.0, 0.0], [0.0, -8.0]], dtype=gs.complex128),
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def frobenius_product_test_data(self):
        smoke_data = [
            dict(
                mat_a=[
                    gs.array([[1.0, -2.0], [1.0, 4.0]], dtype=gs.complex128),
                    gs.array([[1.0, 2.0], [0.0, -3.0]], dtype=gs.complex128),
                ],
                mat_b=[
                    gs.array([[0.0, 4.0], [2.0, 4.0]], dtype=gs.complex128),
                    gs.array([[1.0, -1.0], [5.0, 4.0]], dtype=gs.complex128),
                ],
                expected=[10.0, -13.0],
            ),
            dict(
                mat_a=gs.array([[5.0, 8.0], [2.0, 2.0]], dtype=gs.complex128),
                mat_b=gs.array([[0.0, 0.25], [0.5, 2.0]], dtype=gs.complex128),
                expected=7.0,
            ),
        ]
        return self.generate_tests(smoke_data)

    def trace_product_test_data(self):
        smoke_data = [
            dict(
                mat_a=gs.array([[-2.0, 0.0], [1.0, 2.0]], dtype=gs.complex128),
                mat_b=gs.array([[0.0, 1.0], [2.0, -2.0]], dtype=gs.complex128),
                expected=-3.0 + 0j,
            ),
            dict(
                mat_a=[
                    gs.array([[-5.0, 0.0], [-2.0, 0.0]], dtype=gs.complex128),
                    gs.array([[-2.0, 1.0], [-5.0, -6.0]], dtype=gs.complex128),
                ],
                mat_b=[
                    gs.array([[6.0, 5.0], [-3.0, -2.0]], dtype=gs.complex128),
                    gs.array([[-2.0, 0.0], [4.0, -6.0]], dtype=gs.complex128),
                ],
                expected=gs.array([-40.0, 44.0], dtype=gs.complex128),
            ),
        ]
        return self.generate_tests(smoke_data)

    def flatten_test_data(self):
        smoke_data = [
            dict(
                m=1,
                n=1,
                mat=gs.array([[1.0]], dtype=gs.complex128),
                expected=gs.array([1.0], dtype=gs.complex128),
            ),
            dict(
                m=2,
                n=2,
                mat=EYE_2,
                expected=gs.array([1.0, 0.0, 0.0, 1.0], dtype=gs.complex128),
            ),
            dict(
                m=2,
                n=3,
                mat=MAT1_23,
                expected=gs.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=gs.complex128),
            ),
            dict(
                m=2,
                n=2,
                mat=[EYE_2, MINUS_EYE_2],
                expected=[
                    gs.array([1.0, 0.0, 0.0, 1.0], dtype=gs.complex128),
                    gs.array([-1.0, 0.0, 0.0, -1.0], dtype=gs.complex128),
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def flatten_reshape_test_data(self):
        random_data = [
            dict(m=1, n=1, mat=ComplexMatrices(1, 1).random_point(10000)),
            dict(m=2, n=2, mat=ComplexMatrices(2, 2).random_point(1000)),
            dict(m=2, n=10, mat=ComplexMatrices(2, 10).random_point(100)),
            dict(m=20, n=10, mat=ComplexMatrices(20, 10).random_point(100)),
        ]
        return self.generate_tests([], random_data)

    def diagonal_test_data(self):
        smoke_data = [
            dict(
                m=2, n=2, mat=EYE_2, expected=gs.array([1.0, 1.0], dtype=gs.complex128)
            ),
            dict(
                m=3,
                n=3,
                mat=[MAT1_33, MAT3_33],
                expected=[
                    gs.array([1.0, 4.0, 6.0], dtype=gs.complex128),
                    gs.array([0.0, 0.0, 0.0], dtype=gs.complex128),
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def transpose_test_data(self):
        transpose_MAT3_33 = gs.array(
            [[0.0, -1.0, 2.0], [1.0, 0.0, 3.0], [-2.0, -3.0, 0.0]], dtype=gs.complex128
        )
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

    def transconjugate_test_data(self):
        smoke_data = [
            dict(m=3, n=3, mat=EYE_3, expected=EYE_3),
            dict(
                m=3,
                n=3,
                mat=[MAT9_33, MAT11_33],
                expected=[MAT9_33, MAT12_33],
            ),
        ]
        return self.generate_tests(smoke_data)

    def is_diagonal_test_data(self):
        smoke_data = [
            dict(m=1, n=1, mat=MAT1_11, expected=True),
            dict(m=2, n=2, mat=EYE_2, expected=True),
            dict(m=2, n=3, mat=MAT1_23, expected=False),
            dict(
                m=3,
                n=3,
                mat=[
                    EYE_3,
                    MAT3_33,
                    MAT4_33,
                    MAT5_33,
                    MAT6_33,
                    MAT7_33,
                    MAT8_33,
                    MAT9_33,
                ],
                expected=[True, False, False, False, False, False, False, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def is_symmetric_test_data(self):
        smoke_data = [
            dict(m=1, n=1, mat=MAT1_11, expected=True),
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

    def is_hermitian_test_data(self):
        smoke_data = [
            dict(m=1, n=1, mat=MAT1_11, expected=True),
            dict(m=2, n=2, mat=EYE_2, expected=True),
            dict(m=2, n=3, mat=MAT1_23, expected=False),
            dict(
                m=3,
                n=3,
                mat=[MAT9_33, MAT10_33, MAT11_33],
                expected=[True, True, False],
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
            dict(
                m=3, n=3, mat=[MAT2_33, MAT3_33, MAT9_33], expected=[False, False, True]
            ),
        ]
        return self.generate_tests(smoke_data)

    def is_spd_test_data(self):
        smoke_data = [
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

    def is_hpd_test_data(self):
        smoke_data = [
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
                mat=[MAT1_33, MAT2_33, MAT3_33, MAT4_33, MAT9_33, MAT10_33, MAT11_33],
                expected=[False, False, False, True, True, False, False],
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
                mat=gs.array([[1.0, 2.0], [3.0, 4.0]], dtype=gs.complex128),
                expected=gs.array([[1.0, 0.0], [0.0, 4.0]], dtype=gs.complex128),
            ),
            dict(
                m=2,
                n=2,
                mat=[
                    gs.array([[1.0, 2.0], [3.0, 4.0]], dtype=gs.complex128),
                    gs.array([[5.0, 6.0], [7.0, 8.0]], dtype=gs.complex128),
                ],
                expected=[
                    gs.array([[1.0, 0.0], [0.0, 4.0]], dtype=gs.complex128),
                    gs.array([[5.0, 0.0], [0.0, 8.0]], dtype=gs.complex128),
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_symmetric_test_data(self):
        res = 0.5 * (1e100 + 1e-100)
        smoke_data = [
            dict(
                m=2,
                n=2,
                mat=MAT1_22,
                expected=MAT1_22,
            ),
            dict(
                m=3,
                n=3,
                mat=[
                    gs.array(
                        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [3.0, 1.0, 1.0]],
                        dtype=gs.complex128,
                    ),
                    gs.array(
                        [
                            [1e100, 1e-100, 1e100],
                            [1e100, 1e-100, 1e100],
                            [1e-100, 1e-100, 1e100],
                        ],
                        dtype=gs.complex128,
                    ),
                ],
                expected=[
                    gs.array(
                        [[1.0, 1.0, 3.0], [1.0, 0.0, 0.5], [3.0, 0.5, 1.0]],
                        dtype=gs.complex128,
                    ),
                    gs.array(
                        [[1e100, res, res], [res, 1e-100, res], [res, res, 1e100]],
                        dtype=gs.complex128,
                    ),
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_hermitian_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                mat=MAT1_22,
                expected=MAT1_22,
            ),
            dict(
                m=3,
                n=3,
                mat=[MAT10_33, MAT11_33],
                expected=[MAT10_33, MAT13_33],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_lower_triangular_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                mat=gs.array([[1.0, 2.0], [3.0, 4.0]], dtype=gs.complex128),
                expected=gs.array([[1.0, 0.0], [3.0, 4.0]], dtype=gs.complex128),
            ),
            dict(
                m=2,
                n=2,
                mat=[
                    gs.array([[1.0, 2.0], [3.0, 4.0]], dtype=gs.complex128),
                    gs.array([[5.0, 6.0], [7.0, 8.0]], dtype=gs.complex128),
                ],
                expected=[
                    gs.array([[1.0, 0.0], [3.0, 4.0]], dtype=gs.complex128),
                    gs.array([[5.0, 0.0], [7.0, 8.0]], dtype=gs.complex128),
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_upper_triangular_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                mat=gs.array([[1.0, 2.0], [3.0, 4.0]], dtype=gs.complex128),
                expected=gs.array([[1.0, 2.0], [0.0, 4.0]], dtype=gs.complex128),
            ),
            dict(
                m=2,
                n=2,
                mat=[
                    gs.array([[1.0, 2.0], [3.0, 4.0]], dtype=gs.complex128),
                    gs.array([[5.0, 6.0], [7.0, 8.0]], dtype=gs.complex128),
                ],
                expected=[
                    gs.array([[1.0, 2.0], [0.0, 4.0]], dtype=gs.complex128),
                    gs.array([[5.0, 6.0], [0.0, 8.0]], dtype=gs.complex128),
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_strictly_lower_triangular_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                mat=gs.array([[1.0, 2.0], [3.0, 4.0]], dtype=gs.complex128),
                expected=gs.array([[0.0, 0.0], [3.0, 0.0]], dtype=gs.complex128),
            ),
            dict(
                m=2,
                n=2,
                mat=[
                    gs.array([[1.0, 2.0], [3.0, 4.0]], dtype=gs.complex128),
                    gs.array([[5.0, 6.0], [7.0, 8.0]], dtype=gs.complex128),
                ],
                expected=[
                    gs.array([[0.0, 0.0], [3.0, 0.0]], dtype=gs.complex128),
                    gs.array([[0.0, 0.0], [7.0, 0.0]], dtype=gs.complex128),
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_strictly_upper_triangular_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                mat=gs.array([[1.0, 2.0], [3.0, 4.0]], dtype=gs.complex128),
                expected=gs.array([[0.0, 2.0], [0.0, 0.0]], dtype=gs.complex128),
            ),
            dict(
                m=2,
                n=2,
                mat=[
                    gs.array([[1.0, 2.0], [3.0, 4.0]], dtype=gs.complex128),
                    gs.array([[5.0, 6.0], [7.0, 8.0]], dtype=gs.complex128),
                ],
                expected=[
                    gs.array([[0.0, 2.0], [0.0, 0.0]], dtype=gs.complex128),
                    gs.array([[0.0, 6.0], [0.0, 0.0]], dtype=gs.complex128),
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_lower_triangular_diagonal_scaled_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                mat=gs.array([[2.0, 2.0], [3.0, 4.0]], dtype=gs.complex128),
                expected=gs.array([[1.0, 0.0], [3.0, 2.0]], dtype=gs.complex128),
            ),
            dict(
                m=2,
                n=2,
                mat=[
                    gs.array([[2.0, 2.0], [3.0, 4.0]], dtype=gs.complex128),
                    gs.array([[6.0, 6.0], [7.0, 8.0]], dtype=gs.complex128),
                ],
                expected=[
                    gs.array([[1.0, 0], [3.0, 2.0]], dtype=gs.complex128),
                    gs.array([[3.0, 0.0], [7.0, 4.0]], dtype=gs.complex128),
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_matrix_type_is_matrix_type_test_data(self):
        matrix_types = [
            "diagonal",
            "symmetric",
            "hermitian",
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
                mat = gs.cast(
                    gs.random.normal(size=(n_samples, n, n)), dtype=gs.complex128
                )
                mat += 1j * gs.cast(
                    gs.random.normal(size=(n_samples, n, n)), dtype=gs.complex128
                )
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
            ComplexMatrices,
            self.space_args_list,
            self.shape_list,
            self.n_vecs_list,
            is_tangent_atol,
        )

    def to_tangent_is_projection_test_data(self):
        return self._to_tangent_is_projection_test_data(
            ComplexMatrices,
            self.space_args_list,
            self.shape_list,
            self.n_vecs_list,
        )

    def random_point_is_tangent_test_data(self):
        return self._random_point_is_tangent_test_data(
            self.space_args_list, self.n_points_list
        )

    def basis_test_data(self):
        smoke_data = [
            dict(
                n=2,
                m=2,
                expected=gs.array(
                    [
                        gs.cast(
                            gs.array_from_sparse([(i, j)], [1], (2, 2)),
                            dtype=gs.complex128,
                        )
                        for i in range(2)
                        for j in range(2)
                    ]
                    + [
                        1j
                        * gs.cast(
                            gs.array_from_sparse([(i, j)], [1], (2, 2)),
                            dtype=gs.complex128,
                        )
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
                        gs.cast(
                            gs.array_from_sparse([(i, j)], [1], (2, 3)),
                            dtype=gs.complex128,
                        )
                        for i in range(2)
                        for j in range(3)
                    ]
                    + [
                        1j
                        * gs.cast(
                            gs.array_from_sparse([(i, j)], [1], (2, 3)),
                            dtype=gs.complex128,
                        )
                        for i in range(2)
                        for j in range(3)
                    ]
                ),
            ),
        ]
        return self.generate_tests(smoke_data)

    def random_tangent_vec_is_tangent_test_data(self):
        return self._random_tangent_vec_is_tangent_test_data(
            ComplexMatrices, self.space_args_list, self.n_vecs_list
        )


class ComplexMatricesMetricTestData(_RiemannianMetricTestData):
    m_list = random.sample(range(3, 5), 2)
    n_list = random.sample(range(3, 5), 2)
    metric_args_list = list(zip(m_list, n_list))
    space_args_list = metric_args_list
    shape_list = space_args_list
    space_list = [ComplexMatrices(m, n) for m, n in metric_args_list]
    n_points_list = random.sample(range(1, 7), 2)
    n_tangent_vecs_list = random.sample(range(1, 7), 2)
    n_points_a_list = random.sample(range(1, 7), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                tangent_vec_a=gs.array(
                    [[-3.0, 1.0], [-1.0, -2.0]], dtype=gs.complex128
                ),
                tangent_vec_b=gs.array([[-9.0, 0.0], [4.0, 2.0]], dtype=gs.complex128),
                expected=19.0 + 0j,
            ),
            dict(
                m=2,
                n=2,
                tangent_vec_a=[
                    gs.array([[-1.5, 0.0], [2.0, -3.0]], dtype=gs.complex128),
                    gs.array([[0.5, 7.0], [0.5, -2.0]], dtype=gs.complex128),
                ],
                tangent_vec_b=[
                    gs.array([[2.0, 0.0], [2.0, -3.0]], dtype=gs.complex128),
                    gs.array([[-1.0, 0.0], [1.0, -2.0]], dtype=gs.complex128),
                ],
                expected=gs.array([10.0, 4.0], dtype=gs.complex128),
            ),
        ]
        return self.generate_tests(smoke_data)

    def norm_test_data(self):
        smoke_data = [
            dict(
                m=2,
                n=2,
                vector=gs.array([[1.0, 0.0], [0.0, 1.0]], dtype=gs.complex128),
                expected=SQRT_2,
            ),
            dict(
                m=2,
                n=2,
                vector=[
                    gs.array([[3.0, 0.0], [4.0, 0.0]], dtype=gs.complex128),
                    gs.array([[-3.0, 0.0], [-4.0, 0.0]], dtype=gs.complex128),
                ],
                expected=gs.array([5.0, 5.0], dtype=gs.complex128),
            ),
        ]
        return self.generate_tests(smoke_data)

    def inner_product_norm_test_data(self):
        smoke_data = [
            dict(m=5, n=5, mat=ComplexMatrices(5, 5).random_point(100)),
            dict(m=10, n=10, mat=ComplexMatrices(5, 5).random_point(100)),
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

    def exp_after_log_test_data(self):
        return self._exp_after_log_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            rtol=gs.rtol * 100,
            atol=gs.atol * 10000,
        )

    def log_after_exp_test_data(self):
        return self._log_after_exp_test_data(
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

    def dist_is_symmetric_test_data(self):
        return self._dist_is_symmetric_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_a_list,
            self.n_points_b_list,
        )

    def dist_is_positive_test_data(self):
        return self._dist_is_positive_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_a_list,
            self.n_points_b_list,
        )

    def squared_dist_is_positive_test_data(self):
        return self._squared_dist_is_positive_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_a_list,
            self.n_points_b_list,
        )

    def dist_is_norm_of_log_test_data(self):
        return self._dist_is_norm_of_log_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_a_list,
            self.n_points_b_list,
        )

    def dist_point_to_itself_is_zero_test_data(self):
        return self._dist_point_to_itself_is_zero_test_data(
            self.metric_args_list, self.space_list, self.n_points_list
        )

    def inner_product_is_symmetric_test_data(self):
        return self._inner_product_is_symmetric_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        )

    def triangle_inequality_of_dist_test_data(self):
        return self._triangle_inequality_of_dist_test_data(
            self.metric_args_list, self.space_list, self.n_points_list
        )

    def retraction_lifting_test_data(self):
        return self._log_after_exp_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            rtol=gs.rtol * 100,
            atol=gs.atol * 10000,
        )
