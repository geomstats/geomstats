import random

from geomstats.geometry.heisenberg import HeisenbergVectors
from tests.data_generation import _LieGroupTestData, _VectorSpaceTestData


class HeisenbergVectorsTestData(_LieGroupTestData, _VectorSpaceTestData):
    space_args_list = [()] * 3
    shape_list = [(3,)] * 3
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)
    n_tangent_vecs_list = random.sample(range(2, 5), 2)

    def dimension_test_data(self):
        smoke_data = [dict(expected=3)]
        return self.generate_tests(smoke_data)

    def belongs_test_data(self):
        smoke_data = [
            dict(point=[1.0, 2.0, 3.0, 4], expected=False),
            dict(
                point=[[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]],
                expected=[False, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def is_tangent_test_data(self):
        smoke_data = [
            dict(
                vec=[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
                expected=[False, False],
            )
        ]
        return self.generate_tests(smoke_data)

    def jacobian_translation_test_data(self):
        smoke_data = [
            dict(
                vec=[[1.0, -10.0, 0.2], [-2.0, 100.0, 0.5]],
                expected=[
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 0.5, 1.0]],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-50.0, -1.0, 1.0]],
                ],
            )
        ]
        return self.generate_tests(smoke_data)

    def random_point_belongs_test_data(self):
        smoke_space_args_list = [()] * 2
        smoke_n_points_list = [1, 2]
        return self._random_point_belongs_test_data(
            smoke_space_args_list,
            smoke_n_points_list,
            self.space_args_list,
            self.n_points_list,
        )

    def projection_belongs_test_data(self):
        return self._projection_belongs_test_data(
            self.space_args_list, self.shape_list, self.n_points_list
        )

    def to_tangent_is_tangent_test_data(self):
        return self._to_tangent_is_tangent_test_data(
            HeisenbergVectors,
            self.space_args_list,
            self.shape_list,
            self.n_vecs_list,
        )

    def log_after_exp_test_data(self):
        return self._log_after_exp_test_data(
            HeisenbergVectors,
            self.space_args_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        )

    def exp_after_log_test_data(self):
        return self._exp_after_log_test_data(
            HeisenbergVectors, self.space_args_list, self.n_points_list
        )

    def basis_belongs_test_data(self):
        return self._basis_belongs_test_data(self.space_args_list)

    def basis_cardinality_test_data(self):
        return self._basis_cardinality_test_data(self.space_args_list)

    def random_tangent_vec_is_tangent_test_data(self):
        return self._random_tangent_vec_is_tangent_test_data(
            HeisenbergVectors, self.space_args_list, self.n_vecs_list
        )

    def to_tangent_is_projection_test_data(self):
        return self._to_tangent_is_projection_test_data(
            HeisenbergVectors,
            self.space_args_list,
            self.shape_list,
            self.n_vecs_list,
        )

    def random_point_is_tangent_test_data(self):
        return self._random_point_is_tangent_test_data(
            self.space_args_list, self.n_points_list
        )

    def compose_inverse_point_with_point_is_identity_test_data(self):
        return self._compose_inverse_point_with_point_is_identity_test_data(
            HeisenbergVectors, self.space_args_list, self.n_points_list
        )

    def compose_point_with_inverse_point_is_identity_test_data(self):
        return self._compose_point_with_inverse_point_is_identity_test_data(
            HeisenbergVectors, self.space_args_list, self.n_points_list
        )

    def compose_point_with_identity_is_point_test_data(self):
        return self._compose_point_with_identity_is_point_test_data(
            HeisenbergVectors, self.space_args_list, self.n_points_list
        )

    def compose_identity_with_point_is_point_test_data(self):
        return self._compose_identity_with_point_is_point_test_data(
            HeisenbergVectors, self.space_args_list, self.n_points_list
        )

    def to_tangent_at_identity_belongs_to_lie_algebra_test_data(self):
        return self._to_tangent_at_identity_belongs_to_lie_algebra_test_data(
            self.space_args_list, self.shape_list, self.n_vecs_list
        )
