import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices
from tests.conftest import Parametrizer, TestCase, TestData

EYE_2 = [[1.0, 0], [0.0, 1.0]]
EYE_3 = [[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
MAT_2_3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


class TestMatrices(TestCase, metaclass=Parametrizer):
    class TestDataMatrices(TestData):
        def belongs_data(self):
            sq_mat = EYE_2
            non_sq_mat = MAT_2_3
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
                    mat_1=[MAT_2_3, 2 * MAT_2_3],
                    mat_2=[MAT_2_3, 3 * MAT_2_3],
                    expected=[True, False],
                ),
            ]
            return self.generate_tests(smoke_data)

    def test_belongs(self, m, n, mat, expected):
        self.assertAllClose(Matrices(m, n).belongs(gs.array(mat)), gs.array(expected))

    def test_equal(self, m, n, mat1, mat2, expected):
        self.assertAllClose(
            Matrices(m, n).equal(gs.array(mat1), gs.array(mat2)), gs.array(expected)
        )
