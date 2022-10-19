import random

from geomstats.geometry.hermitian_matrices import HermitianMatrices
from tests.data_generation import _VectorSpaceTestData


class HermitianMatricesTestData(_VectorSpaceTestData):
    """Data class for Testing Hermitian Matrices"""

    space_args_list = [(n,) for n in random.sample(range(2, 5), 2)]
    n_points_list = random.sample(range(1, 5), 2)
    shape_list = [(n, n) for (n,), in zip(space_args_list)]
    n_vecs_list = random.sample(range(1, 5), 2)

    Space = HermitianMatrices

    def belongs_test_data(self):
        smoke_data = [
            dict(n=2, mat=[[1.0, 2.0 + 1j], [2.0 - 1j, 1.0]], expected=True),
            dict(n=2, mat=[[1.0, 1.0], [2.0, 1.0]], expected=False),
            dict(
                n=3,
                mat=[[1.0, 2.0, 3.0 + 1j], [2.0, 4.0, 5.0], [3.0 - 1j, 5.0, 6.0]],
                expected=True,
            ),
            dict(
                n=2,
                mat=[[[1.0, 1j], [-1j, 1.0]], [[1.0 + 1j, -1.0], [0.0, 1.0]]],
                expected=[True, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def basis_test_data(self):
        smoke_data = [
            dict(n=1, basis=[[[1.0]]]),
            dict(
                n=2,
                basis=[
                    [[1.0, 0.0], [0, 0]],
                    [[0, 1.0], [1.0, 0]],
                    [[0, 1j], [-1j, 0]],
                    [[0, 0.0], [0, 1.0]],
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def expm_test_data(self):
        smoke_data = [
            dict(mat=[[0.0, 0.0], [0.0, 0.0]], expected=[[1.0, 0.0], [0.0, 1.0]])
        ]
        return self.generate_tests(smoke_data)

    def powerm_test_data(self):
        smoke_data = [
            dict(
                mat=[[1.0, 2.0], [2.0, 3.0]],
                power=1.0,
                expected=[[1.0, 2.0], [2.0, 3.0]],
            ),
            dict(
                mat=[[1.0, 2.0], [2.0, 3.0]],
                power=2.0,
                expected=[[5.0, 8.0], [8.0, 13.0]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def dim_test_data(self):

        smoke_data = [
            dict(n=1, expected_dim=1),
            dict(n=2, expected_dim=4),
            dict(n=5, expected_dim=25),
        ]

        return self.generate_tests(smoke_data, [])

    def to_vector_test_data(self):
        smoke_data = [
            dict(n=1, mat=[[1.0]], expected=[1.0]),
            dict(
                n=3,
                mat=[[1.0, 2.0, 3.0 + 1j], [2.0, 4.0, 5.0], [3.0 - 1j, 5.0, 6.0]],
                expected=[1.0, 2.0, 3.0 + 1j, 4.0, 5.0, 6.0],
            ),
            dict(
                n=3,
                mat=[
                    [[1.0, 2.0, 3.0 + 1j], [2.0, 4.0, 5.0], [3.0 - 1j, 5.0, 6.0]],
                    [[7.0, 8.0 - 2j, 9.0], [8.0 + 2j, 10.0, 11.0], [9.0, 11.0, 12.0]],
                ],
                expected=[
                    [1.0, 2.0, 3.0 + 1j, 4.0, 5.0, 6.0],
                    [7.0, 8.0 - 2j, 9.0, 10.0, 11.0, 12.0],
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def from_vector_test_data(self):
        smoke_data = [
            dict(n=1, vec=[1.0], expected=[[1.0]]),
            dict(
                n=3,
                vec=[1.0, 2.0, 3.0 + 1j, 4.0, 5.0, 6.0],
                expected=[[1.0, 2.0, 3.0 + 1j], [2.0, 4.0, 5.0], [3.0 - 1j, 5.0, 6.0]],
            ),
            dict(
                n=3,
                vec=[
                    [1.0, 2.0, 3.0 + 1j, 4.0, 5.0, 6.0],
                    [7.0, 8.0 - 2j, 9.0, 10.0, 11.0, 12.0],
                ],
                expected=[
                    [[1.0, 2.0, 3.0 + 1j], [2.0, 4.0, 5.0], [3.0 - 1j, 5.0, 6.0]],
                    [[7.0, 8.0 - 2j, 9.0], [8.0 + 2j, 10.0, 11.0], [9.0, 11.0, 12.0]],
                ],
            ),
        ]
        return self.generate_tests(smoke_data)
