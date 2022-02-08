import pytest

import geomstats.backend as gs


class TestData:
    """Class for TestData objects."""

    def generate_tests(self, smoke_test_data, random_test_data=[]):
        """Wrap test data with corresponding markers.

        Parameters
        ----------
        smoke_test_data : list
            Test data that will be marked as smoke.

        random_test_data : list
            Test data that will be marked as random.
            Optional, default: []

        Returns
        -------
        _: list
            Tests.
        """
        tests = []
        if smoke_test_data:
            smoke_tests = [
                pytest.param(*data.values(), marks=pytest.mark.smoke)
                for data in smoke_test_data
            ]
            tests += smoke_tests
        if random_test_data:
            random_tests = [
                pytest.param(*data.values(), marks=pytest.mark.random)
                if isinstance(data, dict)
                else pytest.param(*data, marks=pytest.mark.random)
                for data in random_test_data
            ]
            tests += random_tests
        return tests


class ManifoldTestData(TestData):
    def _projection_belongs_data(self, space_args, shapes, belongs_atol=gs.atol):
        random_data = [
            dict(
                space_args=space_args,
                data=gs.random.normal(size=shape),
                belongs_atol=belongs_atol,
            )
            for shape in shapes
        ]
        return self.generate_tests([], random_data)

    def _to_tangent_is_tangent_data(self, space_args, tangent_shapes):
        random_data = [
            dict(space_args=space_args, data=gs.random.normal(size=tangent_shape))
            for tangent_shape in tangent_shapes
        ]
        return self.generate_tests([], random_data)


class OpenSetTestData(ManifoldTestData):
    def _to_tangent_belongs_ambient_space_data(
        self,
        space_args,
        tangent_shapes,
    ):
        random_data = [
            dict(space_args=space_args, data=gs.random.normal(size=tangent_shape))
            for tangent_shape in tangent_shapes
        ]
        return self.generate_tests([], random_data)


class LieGroupTestData:
    def _exp_log_composition_data(self, group_args, group_cls, n_samples, base_point):
        random_data = [
            dict(
                group_args=group_args,
                tangent_vec=group_cls(group_args).random_tangent_vec(n_samples),
                base_point=group_cls(group_args).random_point(n_samples),
            )
        ]
        return self.generate_tests([], random_data)

    def _log_exp_composition_data(self, group_args, group_cls, n_samples, base_point):
        random_data = [
            dict(
                group_args=group_args,
                tangent_vec=group_cls(group_args).random_tangent_vec(n_samples),
                base_point=group_cls(group_args).random_point(n_samples),
            )
        ]
        return self.generate_tests([], random_data)


class VectorSpaceTestData:
    def _basis_belongs_data(self, space_args, belongs_atol):
        random_data = [dict(space_args=space_args, belongs_atol=belongs_atol)]
        return self.generate_tests([], random_data)

    def _basis_cardinality_data(self, space_args):
        random_data = [dict(space_args=space_args)]
        return self.generate_tests([], random_data)


class LieAlgebraTestData(VectorSpaceTestData):
    def basis_representation_matrix_representation_composition_data(
        self, space_args, space_cls, n_samples, rtol=gs.rtol, atol=gs.atol
    ):
        random_data = [
            dict(
                space_args=space_args,
                matrix_rep=space_cls(*space_args).random_point(n_samples),
                rtol=rtol,
                atol=atol,
            )
        ]
        return self.generate_tests([], random_data)

    def matrix_representation_basis_representation_composition_data(
        self, space_args, space_cls, n_samples, rtol=gs.rtol, atol=gs.atol
    ):
        random_data = [
            dict(
                space_args=space_args,
                basis_rep=space_cls(*space_args).basis_representation(
                    space_cls(*space_args).random_point(n_samples)
                ),
                rtol=rtol,
                atol=atol,
            )
        ]
        return self.generate_tests([], random_data)


class LevelSetTestData(ManifoldTestData):
    def _extrinsic_intrinsic_composition_data(self, space_args):
        random_data = [dict(space_args=space_args)]
        return self.generate_tests([], random_data)

    def _intrinsic_extrinsic_composition_data(self, space_args):
        random_data = [dict(space_args=space_args)]
        return self.generate_tests([], random_data)


class ConnectionTestData(TestData):
    def _exp_belongs_data(
        self, connection_args_list, spaces, n_tangents_list, n_base_points_list
    ):
        random_data = []
        for connection_args, space, n_tangents, n_base_points in zip(
            connection_args_list, spaces, n_tangents_list, n_base_points_list
        ):
            base_point = space.random_point(n_base_points)
            tangent_vec = space.random_tangent_vec(n_tangents, base_point)
            random_data.append(
                [
                    dict(
                        connection_args=connection_args,
                        space=space,
                        tangent_vec=tangent_vec,
                        base_point=base_point,
                    )
                ]
            )
        self.generate_tests([], random_data)

    def _log_is_tangent_data(
        self, connection_args_list, spaces, n_points_list, n_base_points_list
    ):
        random_data = []
        for connection_args, space, n_points, n_base_points in zip(
            connection_args_list, spaces, n_points_list, n_base_points_list
        ):
            point = space.random_point(n_points)
            base_point = space.random_point(n_base_points)
            random_data.append(
                [
                    dict(
                        connection_args=connection_args,
                        space=space,
                        point=point,
                        base_point=base_point,
                    )
                ]
            )
        self.generate_tests([], random_data)

    def _geodesic_ivp_belongs_data(
        self, connection_args_list, spaces, n_initial_points_list, n_tangent_vecs_list
    ):
        random_data = []
        for connection_args, space, n_points, n_tangent_vecs in zip(
            connection_args_list, spaces, n_initial_points_list, n_tangent_vecs_list
        ):
            initial_point = space.random_point(n_points)
            base_point = space.random_tangent_vec(n_tangent_vecs, initial_point)
            random_data.append(
                [
                    dict(
                        connection_args=connection_args,
                        space=space,
                        initial_point=initial_point,
                        base_point=base_point,
                    )
                ]
            )
        self.generate_tests([], random_data)

    def _geodesic_bvp_belongs_data(
        self, connection_args_list, spaces, n_initial_points_list, n_end_points_list
    ):
        random_data = []
        for connection_args, space, n_initial_points, n_end_points in zip(
            connection_args_list, spaces, n_initial_points_list, n_end_points_list
        ):
            initial_point = space.random_point(n_initial_points)
            end_point = space.random_point(n_end_points)
            random_data.append(
                [
                    dict(
                        connection_args=connection_args,
                        space=space,
                        initial_point=initial_point,
                        end_point=end_point,
                    )
                ]
            )
        self.generate_tests([], random_data)

    def _log_exp_composition_data(
        self, connection_args_list, spaces, n_tangents_list, n_base_points_list
    ):
        return self._exp_belongs_data(
            connection_args_list, spaces, n_tangents_list, n_base_points_list
        )

    def _exp_log_composition_data(
        self, connection_args_list, spaces, n_tangents_list, n_base_points_list
    ):
        return self._exp_belongs_data(
            connection_args_list, spaces, n_tangents_list, n_base_points_list
        )


class RiemannianMetricTestData(TestData):
    def _squared_dist_is_symmetric_data(
        self, metric_args_list, metric_cls, spaces, n_points_a_list, n_points_b_list
    ):
        random_data = []
        for metric_args, space, n_points_a, n_points_b in zip(
            metric_args_list, spaces, n_points_a_list, n_points_b_list
        ):
            point_a = spaces.random_point(n_points_a)
            point_b = spaces.random_point(n_points_b)
            random_data.append(
                [dict(metric_args=metric_args, point_a=point_a, point_b=point_b)]
            )
            return self.generate_tests([], random_data)
