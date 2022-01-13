import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from tests.conftest import Parametrizer, TestCase, TestData

EYE_2 = [[1.0, 0], [0.0, 1.0]]
EYE_3 = [[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
MAT_23 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
MAT1_33 = [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]
MAT2_33 = [[1.0, 2.0, 3.0], [2.0, 4.0, 7.0], [3.0, 5.0, 6.0]]
MAT3_33 = [[0.0, 1.0, -2.0], [-1.0, 0.0, -3.0], [2.0, 3.0, 0.0]]
MAT4_33 = [[2.0, -1.0, 0.0], [-1.0, 2.0, 1.0], [0.0, -1.0, 2.0]]
MAT5_33 = [[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [8.0, -1.0, 2.0]]
MAT6_33 = [[1.0, 3.0, 4.0], [0.0, 2.0, 6.0], [0.0, 0.0, 2.0]]
MAT7_33 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [8.0, -1.0, 0.0]]
MAT8_33 = [[0.0, 3.0, 4.0], [0.0, 0.0, 6.0], [0.0, 0.0, 0.0]]


class TestMatrices(TestCase, metaclass=Parametrizer):
    class TestDataMatrices(TestData):
        def belongs_data(self):
            sq_mat = EYE_2
            non_sq_mat = MAT_23
            smoke_data = [
                dict(m=2, n=2, mat=sq_mat, expected=True),
                dict(m=2, n=1, mat=sq_mat, expected=False),
                dict(m=2, n=3, mat=non_sq_mat, expected=True),
                dict(m=2, n=1, mat=non_sq_mat, expected=False),
                dict(m=2, n=3, mat=[non_sq_mat, non_sq_mat], expected=[True, True]),
                dict(m=2, n=3, mat=[non_sq_mat, sq_mat], expected=[True, False]),
            ]
            return self.generate_tests(smoke_data)

        def equal_data(self):

            smoke_data = [
                dict(m=2, n=2, mat_1=EYE_2, mat_2=EYE_2, expected=True),
                dict(m=2, n=2, mat_1=EYE_2, mat_2=2 * EYE_2, expected=False),
                dict(
                    m=2,
                    n=3,
                    mat_1=[MAT_23, 2 * MAT_23],
                    mat_2=[MAT_23, 3 * MAT_23],
                    expected=[True, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def mul(self):
            mats_1 = (
                [[1.0, 2.0], [3.0, 4.0]],
                [[-1.0, 2.0], [-3.0, 4.0]],
                [[1.0, -2.0], [3.0, -4.0]],
            )
            mats_2 = [[[2.0], [4.0]], [[1.0], [3.0]], [[1.0], [3.0]]]
            mat_1_X_mat_2 = [[[10.0], [22.0]], [[5.0], [9.0]], [[-5.0], [-9.0]]]
            smoke_data = [
                dict(mat=mats_1, expected=[[23.0, -26.0], [51.0, -58.0]]),
                dict(mat=(list(mats_1), mats_2), expected=mat_1_X_mat_2),
            ]
            return self.generate_tests(smoke_data)

        def commutator_data(self):
            smoke_data = []
            return self.generate_tests(smoke_data)

        def is_symmetric_data(self):
            smoke_data = [
                dict(m=2, n=2, mat=EYE_2, expected=True),
                dict(m=2, n=3, mat=MAT_23, expected=False),
                dict(m=2, n=2, mat=[EYE_2, EYE_2 + 1], expected=[True, True]),
                dict(
                    m=3,
                    n=3,
                    mat=[MAT1_33, MAT2_33, MAT3_33],
                    expected=[True, False, True],
                ),
            ]
            self.generate_tests(smoke_data)

        def is_skew_symmetric_data(self):
            smoke_data = [
                dict(m=2, n=2, mat=EYE_2, expected=False),
                dict(m=2, n=3, mat=MAT_23, expected=False),
                dict(m=2, n=2, mat=[EYE_2, -1 * EYE_2], expected=[False, False]),
                dict(m=3, n=3, mat=[MAT2_33, MAT3_33], expected=[False, True]),
            ]
            self.generate_tests(smoke_data)

        def is_pd(self):
            smoke_data = [
                dict(m=2, n=2, mat=EYE_2, expected=True),
                dict(m=2, n=3, mat=MAT_23, expected=False),
                dict(m=2, n=2, mat=[EYE_2, -1 * EYE_2], expected=[True, False]),
                dict(m=3, n=3, mat=[MAT2_33, MAT3_33], expected=[False, False]),
                dict(m=3, n=3, mat=[MAT2_33, MAT3_33], expected=[False, False]),
            ]
            self.generate_tests(smoke_data)

        def is_spd(self):
            smoke_data = [
                dict(m=3, n=2, mat=EYE_2, expected=True),
                dict(m=3, n=3, mat=MAT4_33, expected=True),
                dict(m=2, n=3, mat=MAT_23, expected=False),
                dict(
                    m=2,
                    n=2,
                    mat=[EYE_2, -1 * EYE_2, -2 * MAT4_33],
                    expected=[True, False, False],
                ),
                dict(
                    m=3,
                    n=3,
                    mat=[MAT1_33, MAT2_33, MAT3_33],
                    expected=[False, False, False],
                ),
            ]
            self.generate_tests(smoke_data)

        def is_upper_triangularr(self):
            smoke_data = [
                dict(m=2, n=2, mat=EYE_2, expected=True),
                dict(m=2, n=3, mat=MAT_23, expected=False),
                dict(m=3, n=3, mat=MAT6_33, expected=True),
                dict(
                    m=3,
                    n=3,
                    mat=[MAT1_33, MAT2_33, MAT3_33, MAT4_33, EYE_3],
                    expected=[False, False, False, False, True],
                ),
            ]
            return self.generate_tests(smoke_data)

        def is_lower_triangular(self):
            smoke_data = [
                dict(m=2, n=2, mat=EYE_2, expected=True),
                dict(m=2, n=3, mat=MAT_23, expected=False),
                dict(m=3, n=3, mat=MAT5_33, expected=False),
                dict(
                    m=3,
                    n=3,
                    mat=[MAT1_33, MAT2_33, MAT3_33, MAT4_33, EYE_3],
                    expected=[False, False, False, False, True],
                ),
            ]
            return self.generate_tests(smoke_data)

        def is_strictly_lower_triangular(self):
            smoke_data = [
                dict(m=2, n=2, mat=EYE_2, expected=False),
                dict(m=2, n=3, mat=MAT_23, expected=False),
                dict(m=3, n=3, mat=MAT8_33, expected=True),
                dict(
                    m=3,
                    n=3,
                    mat=[MAT1_33, MAT2_33, MAT3_33, MAT4_33, MAT5_33, MAT6_33, EYE_3],
                    expected=[False, False, False, False, False, False, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def is_strictly_upper_triangular(self):
            smoke_data = [
                dict(m=2, n=2, mat=EYE_2, expected=False),
                dict(m=2, n=3, mat=MAT_23, expected=False),
                dict(m=3, n=3, mat=MAT7_33, expected=True),
                dict(
                    m=3,
                    n=3,
                    mat=[MAT1_33, MAT2_33, MAT3_33, MAT4_33, MAT5_33, MAT6_33, EYE_3],
                    expected=[False, False, False, False, False, False, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def to_diagonal_data(self):
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

        def to_lower_triangular_data(self):
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

        def to_upper_triangular_data(self):
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

        def to_strictly_lower_triangular_data(self):
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

        def to_strictly_upper_triangular_data(self):
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

        def to_lower_triangular_diagonal_scaled_data(self):
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

    testing_data = TestDataMatrices()

    def test_belongs(self, m, n, mat, expected):
        self.assertAllClose(Matrices(m, n).belongs(gs.array(mat)), gs.array(expected))

    def test_equal(self, m, n, mat1, mat2, expected):
        self.assertAllClose(
            Matrices(m, n).equal(gs.array(mat1), gs.array(mat2)), gs.array(expected)
        )

    def test_mul(self, mat, expected):
        self.assertAllClose(Matrices.mul(mat), expected)

    def test_commutator(self, mat_a, mat_b, expected):
        self.assertAllClose(Matrices.commutator(mat_a, mat_b), expected)

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

    def test_is_upper_triangualr(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).is_upper_triangular(gs.array(mat)), gs.array(expected)
        )

    def test_is_lower_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).is_lower_triangular(gs.array(mat)), gs.array(expected)
        )

    def test_is_strictly_lower_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            Matrices(m, n).is_strictly_upper_triangular(gs.array(mat)),
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
