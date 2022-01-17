"""Unit tests for the vector space of lower triangular matrices."""

import geomstats.backend as gs
from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices
from tests.conftest import Parametrizer, TestCase, TestData


class TestLowerTriangularMatrices(TestCase, metaclass=Parametrizer):
    """Test of LowerTriangularMatrices methods."""

    space = LowerTriangularMatrices

    class TestDataLowerTriangularMatrices(TestData):
        def belongs_data(self):
            smoke_data = [
                dict(n=2, mat=[[1.0, 0.0], [-1.0, 3.0]], expected=True),
                dict(n=2, mat=[[1.0, -1.0], [-1.0, 3.0]], expected=False),
                dict(
                    n=2,
                    mat=[
                        [[1.0, 0], [0, 1.0]],
                        [[1.0, 2.0], [2.0, 1.0]],
                        [[-1.0, 0.0], [1.0, 1.0]],
                        [[0.0, 0.0], [1.0, 1.0]],
                    ],
                    expected=[True, False, True, True],
                ),
                dict(
                    n=3,
                    mat=[
                        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[-1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    ],
                    expected=[False, True, True, True],
                ),
                dict(n=3, mat=[[1.0, 0.0], [-1.0, 3.0]], expected=False),
            ]
            return self.generate_tests(smoke_data)

        def random_point_and_belongs_data(self):
            smoke_data = [
                dict(n=1, n_points=1),
                dict(n=2, n_points=2),
                dict(n=10, n_points=100),
                dict(n=100, n_points=10),
            ]
            return self.generate_tests(smoke_data)

        def to_vector_data(self):
            smoke_data = [
                dict(
                    n=3,
                    mat=[[1.0, 0.0, 0.0], [0.6, 7.0, 0.0], [-3.0, 0.0, 8.0]],
                    expected=[1.0, 0.6, 7.0, -3.0, 0.0, 8.0],
                ),
                dict(
                    n=3,
                    mat=[
                        [[1.0, 0.0, 0.0], [0.6, 7.0, 0.0], [-3.0, 0.0, 8.0]],
                        [[2.0, 0.0, 0.0], [2.6, 7.0, 0.0], [-3.0, 0.0, 28.0]],
                    ],
                    expected=[
                        [1.0, 0.6, 7.0, -3.0, 0.0, 8.0],
                        [2.0, 2.6, 7.0, -3.0, 0.0, 28.0],
                    ],
                ),
            ]
            return self.generate_tests(smoke_data)

        def get_basis_data(self):
            smoke_data = [
                dict(
                    n=2,
                    expected=[
                        [[1.0, 0.0], [0.0, 0.0]],
                        [[0.0, 0.0], [1.0, 0.0]],
                        [[0.0, 0.0], [0.0, 1.0]],
                    ],
                )
            ]
            return self.generate_tests(smoke_data)

        def projection_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point=[[2.0, 1.0], [1.0, 2.0]],
                    expected=[[2.0, 0.0], [1.0, 2.0]],
                ),
                dict(
                    n=2,
                    point=[[1.0, 0.0], [0.0, 1.0]],
                    expected=[[1.0, 0.0], [0.0, 1.0]],
                ),
            ]
            return self.generate_tests(smoke_data)

    testing_data = TestDataLowerTriangularMatrices()

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(self.space(n).belongs(gs.array(mat)), gs.array(expected))

    def test_random_point_and_belongs(self, n, n_points):
        space_n = self.space(n)
        self.assertAllClose(
            gs.all(space_n.belongs(space_n.random_point(n_points))), True
        )

    def test_to_vector(self, n, mat, expected):
        self.assertAllClose(self.space(n).to_vector(gs.array(mat)), gs.array(expected))

    def test_get_basis(self, n, expected):
        self.assertAllClose(self.space(n).get_basis(), gs.array(expected))

    def test_projection(self, n, point, expected):
        self.assertAllClose(self.space(n).projection(point), gs.array(expected))
