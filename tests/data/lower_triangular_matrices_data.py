import random

from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices
from tests.data_generation import _VectorSpaceTestData


class LowerTriangularMatricesTestData(_VectorSpaceTestData):
    Space = LowerTriangularMatrices

    n_list = random.sample(range(2, 5), 2)
    space_args_list = [(n,) for n in n_list]
    shape_list = [(n, n) for n in n_list]
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    def belongs_test_data(self):
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

    def random_point_and_belongs_test_data(self):
        smoke_data = [
            dict(n=1, n_points=1),
            dict(n=2, n_points=2),
            dict(n=10, n_points=100),
            dict(n=100, n_points=10),
        ]
        return self.generate_tests(smoke_data)

    def to_vector_test_data(self):
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

    def get_basis_test_data(self):
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

    def projection_test_data(self):
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
