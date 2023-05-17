import random

import geomstats.backend as gs
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from tests.data_generation import _VectorSpaceTestData


class SymmetricMatricesTestData(_VectorSpaceTestData):
    """Data class for Testing Symmetric Matrices"""

    Space = SymmetricMatrices

    space_args_list = [(n,) for n in random.sample(range(2, 5), 2)]
    n_points_list = random.sample(range(1, 5), 2)
    shape_list = [(n, n) for (n,), in zip(space_args_list)]
    n_vecs_list = random.sample(range(1, 5), 2)

    def belongs_test_data(self):
        smoke_data = [
            dict(n=2, mat=gs.array([[1.0, 2.0], [2.0, 1.0]]), expected=True),
            dict(n=2, mat=gs.array([[1.0, 1.0], [2.0, 1.0]]), expected=False),
            dict(
                n=3,
                mat=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
                expected=True,
            ),
            dict(
                n=2,
                mat=gs.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [0.0, 1.0]]]),
                expected=[True, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def basis_test_data(self):
        smoke_data = [
            dict(n=1, basis=gs.array([[[1.0]]])),
            dict(
                n=2,
                basis=gs.array(
                    [
                        [[1.0, 0.0], [0, 0]],
                        [[0, 1.0], [1.0, 0]],
                        [[0, 0.0], [0, 1.0]],
                    ]
                ),
            ),
        ]
        return self.generate_tests(smoke_data)

    def expm_test_data(self):
        smoke_data = [
            dict(
                mat=gs.array([[0.0, 0.0], [0.0, 0.0]]),
                expected=gs.array([[1.0, 0.0], [0.0, 1.0]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def powerm_test_data(self):
        smoke_data = [
            dict(
                mat=gs.array([[1.0, 2.0], [2.0, 3.0]]),
                power=1.0,
                expected=gs.array([[1.0, 2.0], [2.0, 3.0]]),
            ),
            dict(
                mat=gs.array([[1.0, 2.0], [2.0, 3.0]]),
                power=2.0,
                expected=gs.array([[5.0, 8.0], [8.0, 13.0]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def dim_test_data(self):

        smoke_data = [
            dict(n=1, expected_dim=1),
            dict(n=2, expected_dim=3),
            dict(n=5, expected_dim=15),
        ]

        return self.generate_tests(smoke_data, [])

    def to_vector_test_data(self):
        smoke_data = [
            dict(n=1, mat=gs.array([[1.0]]), expected=gs.array([1.0])),
            dict(
                n=3,
                mat=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
                expected=gs.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
            dict(
                n=3,
                mat=gs.array(
                    [
                        [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                        [[7.0, 8.0, 9.0], [8.0, 10.0, 11.0], [9.0, 11.0, 12.0]],
                    ]
                ),
                expected=gs.array(
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    ]
                ),
            ),
        ]
        return self.generate_tests(smoke_data)

    def from_vector_test_data(self):
        smoke_data = [
            dict(n=1, vec=gs.array([1.0]), expected=gs.array([[1.0]])),
            dict(
                n=3,
                vec=gs.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                expected=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
            ),
            dict(
                n=3,
                vec=gs.array(
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    ]
                ),
                expected=gs.array(
                    [
                        [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                        [[7.0, 8.0, 9.0], [8.0, 10.0, 11.0], [9.0, 11.0, 12.0]],
                    ]
                ),
            ),
        ]
        return self.generate_tests(smoke_data)
