import pytest

import geomstats.backend as gs
from geomstats.test.data import TestData


class GroupActionTestData(TestData):
    def identity_action_test_data(self):
        return self.generate_random_data()

    def action_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_element_action_test_data(self):
        return self.generate_random_data()


class RowPermutationAction4TestData(TestData):
    """Permutation action data

    Smoke data from
    https://www.sciencedirect.com/topics/engineering/permutation-matrix.
    """

    def action_test_data(self):
        data = [
            dict(
                group_elem=gs.array([1, 2, 3, 0]),
                point=gs.array(
                    [
                        [4, 6, 8, 1],
                        [5, 3, 1, 0],
                        [7, 21, 0, 9],
                        [3, -4, 1, 3],
                    ]
                ),
                expected=gs.array(
                    [
                        [3, -4, 1, 3],
                        [4, 6, 8, 1],
                        [5, 3, 1, 0],
                        [7, 21, 0, 9],
                    ]
                ),
            )
        ]

        return self.generate_tests(data, marks=[pytest.mark.smoke])


class PermutationMatrixFromVectorTestData(TestData):
    """Permutation matrix from vector data.

    Smoke data from
    https://www.sciencedirect.com/topics/engineering/permutation-matrix.
    """

    def permutation_matrix_from_vector_vec_test_data(self):
        return self.generate_vec_data()

    def permutation_matrix_from_vector_test_data(self):
        data = [
            dict(
                group_elem=gs.array([1, 2, 3, 0]),
                expected=gs.array(
                    [
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                    ]
                ),
            )
        ]

        return self.generate_tests(data, marks=[pytest.mark.smoke])
