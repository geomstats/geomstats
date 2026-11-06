import pytest
from polpo.testing.data import CaseData
from polpo.testing.geometric import GeometricCaseData

import geomstats.backend as gs
from geomstats.geometry.group_action import permutation_matrix_from_vector
from geomstats.test.random import GroupActionRandomDataGenerator, RandomDataGenerator


class GroupActionTestData(GeometricCaseData):
    """Group action data.

    Points are sampled from `space` and group elements from `group`.
    """

    def __init__(self, space, group, group_action, **kwargs):
        super().__init__(space, **kwargs)
        self.group = group
        self.group_action = group_action
        self.data_generator = GroupActionRandomDataGenerator(
            space, group, **self.generator_kwargs
        )

    def _get_operation(self, op_name, op_target=None):
        return getattr(self.group_action, op_name)

    def action_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("group_elem", "point"),
            op_name="__call__",
            samplers={"group_elem": "random_group_elem"},
        )

    def identity_action_test_data(self):
        return self.generate_random_data(arg_names="point")

    def inverse_action_test_data(self):
        return self.generate_random_data(
            arg_names=("group_elem", "point"),
            samplers={"group_elem": "random_group_elem"},
        )


class RowPermutationAction4TestData(CaseData):
    """Permutation action data

    Smoke data from
    https://www.sciencedirect.com/topics/engineering/permutation-matrix.
    """

    @pytest.mark.smoke
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

        return data


class PermutationMatrixFromVectorTestData(GeometricCaseData):
    """Permutation matrix from vector data.

    Group elements are sampled from `group`.

    Smoke data from
    https://www.sciencedirect.com/topics/engineering/permutation-matrix.
    """

    def __init__(self, group, **kwargs):
        super().__init__(**kwargs)
        self.data_generator = RandomDataGenerator(group)

    def _get_operation(self, op_name, op_target=None):
        return permutation_matrix_from_vector

    def permutation_matrix_from_vector_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="group_elem",
            samplers={"group_elem": "random_point"},
        )

    @pytest.mark.smoke
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

        return data
