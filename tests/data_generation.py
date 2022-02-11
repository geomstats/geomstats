import itertools
import random

import pytest

import geomstats.backend as gs


def better_squeeze(array):
    if len(array) == 1:
        return gs.squeeze(array, axis=0)
    return array


class TemporaryTestData:
    """Class for TemporaryTestData objects for backward compatibility."""

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

    def _log_exp_composition_data(
        self, space, n_samples=100, max_n=10, n_n=5, **kwargs
    ):
        """Generate Data that checks for log and exp are inverse. Specifically
            :math: `Exp_{base_point}(Log_{base_point}(point)) = point`
        Parameters
        ----------
        space : cls
            Manifold class on which metric is present.
        max_n : int
            Maximum value when generating 'n'.
            Optional, default: 20
        n_n : int
            Number of 'n' to be generated.
            Optional, default: 5
        n_samples : int
            Optional, default: 100
        Returns
        -------
        _ : list
            Test Data.
        """
        random_n = random.sample(range(1, max_n), n_n)
        random_data = []
        for n in random_n:
            for prod in itertools.product(*kwargs.values()):
                space_n = space(n)
                base_point = space_n.random_point(n_samples)
                point = space_n.random_point(n_samples)
                random_data.append((n,) + prod + (point, base_point))
        return self.generate_tests([], random_data)

    def _geodesic_belongs_data(
        self, space, max_n=10, n_n=5, n_geodesics=10, n_t=10, **kwargs
    ):
        """Generate Data that checks for points on geodesic belongs to data.
        Parameters
        ----------
        space : cls
            Manifold class on which metric is present.
        max_n : int
            Maximum value when generating 'n'.
            Optional, default: 10
        n_n : int
            Maximum value when generating 'n'.
            Optional, default: 5
        n_geodesics : int
            Number of geodesics to be generated.
            Optional, default: 10
        n_t : int
            Number of points to be sampled on each geodesic.
            Optional, default: 10
        Returns
        -------
        _ : list
            Test Data.
        """
        random_n = random.sample(range(2, max_n), n_n)
        random_data = []
        for n in random_n:
            for prod in itertools.product(*kwargs.values()):
                space_n = space(n)
                initial_point = space_n.random_point()
                initial_tangent_points = space_n.random_tangent_vec(
                    n_geodesics, base_point=initial_point
                )
                random_t = gs.linspace(start=-1.0, stop=1.0, num=n_t)
                for initial_tangent_point, t in itertools.product(
                    initial_tangent_points, random_t
                ):
                    random_data.append(
                        (n,) + prod + (initial_point, initial_tangent_point, t)
                    )
        return self.generate_tests([], random_data)

    def _squared_dist_is_symmetric_data(
        self, space, max_n=5, n_n=3, n_samples=10, **kwargs
    ):
        """Generate Data that checks squared_dist is symmetric.
        Parameters
        ----------
        space : cls
            Manifold class on which metric is present.
        max_n : int
            Range of 'n' to generated.
            Optional, default: 10
        n_n : int
            Maximum value when generating 'n'.
            Optional, default: 3
        n_samples : int
            Number of points to be generated.
            Optional, default: 10
        Returns
        -------
        _ : list
            Test Data.
        """
        random_n = random.sample(range(2, max_n), n_n)
        random_data = []
        for n in random_n:
            for prod in itertools.product(*kwargs.values()):
                space_n = space(n)
                points_a = space_n.random_point(n_samples)
                points_b = space_n.random_point(n_samples)
                for point_a, point_b in itertools.product(points_a, points_b):
                    random_data.append((n,) + prod + (point_a, point_b))
        return self.generate_tests([], random_data)


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
    def _random_point_belongs_data(
        self,
        smoke_space_args_list,
        smoke_n_points_list,
        space_args_list,
        n_points_list,
        belongs_atol=gs.atol,
    ):

        smoke_data = [
            dict(space_args=space_args, n_points=n_points, belongs_atol=belongs_atol)
            for space_args, n_points in zip(smoke_space_args_list, smoke_n_points_list)
        ]
        random_data = [
            dict(space_args=space_args, n_points=n_points, belongs_atol=belongs_atol)
            for space_args, n_points in zip(space_args_list, n_points_list)
        ]
        return self.generate_tests(smoke_data, random_data)

    def _projection_belongs_data(
        self, space_args_list, shapes_list, n_samples_list, cls, belongs_atol=gs.atol
    ):

        random_data = [
            dict(
                space_args=space_args,
                data=gs.random.normal(size=(n_samples,) + shape),
                cls=cls,
                belongs_atol=belongs_atol,
            )
            for space_args, shape, n_samples in zip(
                space_args_list, shapes_list, n_samples_list
            )
        ]
        return self.generate_tests([], random_data)

    def _to_tangent_is_tangent_data(
        self,
        space_cls,
        space_args_list,
        tangent_shapes_list,
        n_vecs_list,
        is_tangent_atol=gs.atol,
    ):

        random_data = []
        for space_args, tangent_shape, n_vecs in zip(
            space_args_list, tangent_shapes_list, n_vecs_list
        ):
            space = space_cls(*space_args)
            vec = gs.random.normal(size=(n_vecs,) + tangent_shape)
            base_point = space.random_point()
            random_data.append(
                dict(
                    space_args=space_args,
                    vec=vec,
                    base_point=base_point,
                    is_tangent_atol=is_tangent_atol,
                )
            )
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


class LieGroupTestData(ManifoldTestData):
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


class VectorSpaceTestData(ManifoldTestData):
    def _basis_belongs_data(self, space_args_list, belongs_atol=gs.atol):
        random_data = [
            dict(space_args=space_args, belongs_atol=belongs_atol)
            for space_args in space_args_list
        ]
        return self.generate_tests([], random_data)

    def _basis_cardinality_data(self, space_args_list):
        random_data = [dict(space_args=space_args) for space_args in space_args_list]
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
    def _exp_shape_data(
        self, connection_args_list, space_list, tangent_shape_list, batch_size_list
    ):
        random_data = []
        for connection_args, space, tangent_shape, batch_size in zip(
            connection_args_list, space_list, tangent_shape_list, batch_size_list
        ):
            base_point = space.random_point(batch_size)
            tangent_vec = space.to_tangent(
                gs.random.normal(size=(batch_size,) + tangent_shape), base_point
            )
            n_points_list = itertools.product([1, batch_size], [1, batch_size])
            expected_shape_list = [space.shape] + [(batch_size,) + space.shape] * 3
            for (n_tangent_vecs, n_base_points), expected_shape in zip(
                n_points_list, expected_shape_list
            ):
                random_data.append(
                    dict(
                        connection_args=connection_args,
                        tangent_vec=better_squeeze(tangent_vec[:n_tangent_vecs]),
                        base_point=better_squeeze(base_point[:n_base_points]),
                        expected_shape=expected_shape,
                    )
                )
        return self.generate_tests([], random_data)

    def _log_shape_data(self, connection_args_list, space_list, batch_size_list):
        random_data = []
        for connection_args, space, batch_size in zip(
            connection_args_list, space_list, batch_size_list
        ):
            base_point = space.random_point(batch_size)
            point = space.random_point(batch_size)
            n_points_list = itertools.product([1, batch_size], [1, batch_size])
            expected_shape_list = [space.shape] + [(batch_size,) + space.shape] * 3
            for (n_points, n_base_points), expected_shape in zip(
                n_points_list, expected_shape_list
            ):

                random_data.append(
                    dict(
                        connection_args=connection_args,
                        point=better_squeeze(point[:n_points]),
                        base_point=better_squeeze(base_point[:n_base_points]),
                        expected_shape=expected_shape,
                    )
                )
        return self.generate_tests([], random_data)

    def _exp_belongs_data(
        self,
        connection_args_list,
        space_list,
        tangent_shape_list,
        n_tangent_vecs_list,
        belongs_atol=gs.atol,
    ):
        random_data = []
        for connection_args, space, tangent_shape, n_tangent_vecs in zip(
            connection_args_list, space_list, tangent_shape_list, n_tangent_vecs_list
        ):
            base_point = space.random_point()
            tangent_vec = space.to_tangent(
                gs.random.normal(size=(n_tangent_vecs,) + tangent_shape), base_point
            )
            random_data.append(
                dict(
                    connection_args=connection_args,
                    space=space,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    belongs_atol=belongs_atol,
                )
            )
        return self.generate_tests([], random_data)

    def _log_is_tangent_data(
        self, connection_args_list, space_list, n_points_list, is_tangent_atol=gs.atol
    ):
        random_data = []
        for connection_args, space, n_points in zip(
            connection_args_list, space_list, n_points_list
        ):
            point = space.random_point(n_points)
            base_point = space.random_point()
            random_data.append(
                dict(
                    connection_args=connection_args,
                    space=space,
                    point=point,
                    base_point=base_point,
                    is_tangent_atol=is_tangent_atol,
                )
            )
        return self.generate_tests([], random_data)

    def _geodesic_ivp_belongs_data(
        self,
        connection_args_list,
        space_list,
        n_t_list,
        tangent_shapes_list,
        belongs_atol=gs.atol,
    ):
        random_data = []
        for connection_args, space, n_t, tangent_shape in zip(
            connection_args_list, space_list, n_t_list, tangent_shapes_list
        ):
            initial_point = space.random_point()
            initial_tangent_vec = space.to_tangent(
                gs.random.normal(size=tangent_shape), initial_point
            )
            random_data.append(
                dict(
                    connection_args=connection_args,
                    space=space,
                    n_t=n_t,
                    initial_point=initial_point,
                    initial_tangent_vec=initial_tangent_vec,
                    belongs_atol=belongs_atol,
                )
            )
        return self.generate_tests([], random_data)

    def _geodesic_bvp_belongs_data(
        self,
        connection_args_list,
        space_list,
        n_t_list,
        belongs_atol=gs.atol,
    ):
        random_data = []
        for connection_args, space, n_t in zip(
            connection_args_list,
            space_list,
            n_t_list,
        ):
            initial_point = space.random_point()
            end_point = space.random_point()
            random_data.append(
                dict(
                    connection_args=connection_args,
                    space=space,
                    n_t=n_t,
                    initial_point=initial_point,
                    end_point=end_point,
                    belongs_atol=belongs_atol,
                )
            )
        return self.generate_tests([], random_data)

    def _log_exp_composition_data(
        self,
        connection_args_list,
        space_list,
        tangent_shape_list,
        n_tangent_vecs_list,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        random_data = []
        for connection_args, space, tangent_shape, n_tangent_vecs in zip(
            connection_args_list, space_list, tangent_shape_list, n_tangent_vecs_list
        ):
            base_point = space.random_point()
            tangent_vec = space.to_tangent(
                gs.random.normal(size=(n_tangent_vecs,) + tangent_shape), base_point
            )
            random_data.append(
                dict(
                    connection_args=connection_args,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    rtol=rtol,
                    atol=atol,
                )
            )
        return self.generate_tests([], random_data)

    def _exp_log_composition_data(
        self,
        connection_args_list,
        space_list,
        n_points_list,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        random_data = []
        for connection_args, space, n_points in zip(
            connection_args_list, space_list, n_points_list
        ):
            point = space.random_point(n_points)
            base_point = space.random_point()
            random_data.append(
                dict(
                    connection_args=connection_args,
                    point=point,
                    base_point=base_point,
                    rtol=rtol,
                    atol=atol,
                )
            )
        return self.generate_tests([], random_data)

    def _exp_ladder_parallel_transport_data(
        self,
        connection_args_list,
        spaces_list,
        tangent_shape_list,
        n_tangent_vecs_list,
        n_rungs_list,
        alpha_list,
        scheme_list,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        random_data = []
        for (
            connection_args,
            space,
            tangent_shape,
            n_tangent_vecs,
            n_rungs,
            alpha,
            scheme,
        ) in zip(
            connection_args_list,
            spaces_list,
            tangent_shape_list,
            n_tangent_vecs_list,
            n_rungs_list,
            alpha_list,
            scheme_list,
        ):
            base_point = space.random_point()

            tangent_vec = space.to_tangent(
                gs.random.normal(size=(n_tangent_vecs,) + tangent_shape), base_point
            )
            direction = space.to_tangent(
                gs.random.normal(size=tangent_shape), base_point
            )
            random_data.append(
                dict(
                    connection_args=connection_args,
                    direction=direction,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    scheme=scheme,
                    n_rungs=n_rungs,
                    alpha=alpha,
                    rtol=rtol,
                    atol=atol,
                )
            )

        return self.generate_tests([], random_data)

    def _exp_geodesic_ivp_data(
        self,
        connection_args_list,
        spaces_list,
        tangent_shapes_list,
        n_tangent_vecs_list,
        n_points_list,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        random_data = []
        for connection_args, space, tangent_shape, n_tangent_vecs, n_points in zip(
            connection_args_list,
            spaces_list,
            tangent_shapes_list,
            n_tangent_vecs_list,
            n_points_list,
        ):
            base_point = space.random_point()
            tangent_vec = space.to_tangent(
                gs.random.normal(size=(n_tangent_vecs,) + tangent_shape), base_point
            )
            random_data.append(
                dict(
                    connection_args=connection_args,
                    n_points=n_points,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    rtol=rtol,
                    atol=atol,
                )
            )
        return self.generate_tests([], random_data)


class RiemannianMetricTestData(ConnectionTestData):
    def _squared_dist_is_symmetric_data(
        self,
        metric_args_list,
        spaces_list,
        n_points_a_list,
        n_points_b_list,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        random_data = []
        for metric_args, space, n_points_a, n_points_b in zip(
            metric_args_list, spaces_list, n_points_a_list, n_points_b_list
        ):
            point_a = space.random_point(n_points_a)
            point_b = space.random_point(n_points_b)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    point_a=point_a,
                    point_b=point_b,
                    rtol=rtol,
                    atol=atol,
                )
            )
        return self.generate_tests([], random_data)

    def _parallel_transport_ivp_is_isometry_data(
        self,
        metric_args_list,
        space_list,
        tangent_shape_list,
        n_tangent_vecs_list,
        is_tangent_atol=gs.atol,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        random_data = []
        for metric_args, space, tangent_shape, n_tangent_vecs in zip(
            metric_args_list, space_list, tangent_shape_list, n_tangent_vecs_list
        ):
            base_point = space.random_point()

            tangent_vec = space.to_tangent(
                gs.random.normal(size=(n_tangent_vecs,) + tangent_shape), base_point
            )
            direction = space.to_tangent(
                gs.random.normal(size=tangent_shape), base_point
            )
            random_data.append(
                dict(
                    metric_args=metric_args,
                    space=space,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    direction=direction,
                    is_tangent_atol=is_tangent_atol,
                    rtol=rtol,
                    atol=atol,
                )
            )

        return self.generate_tests([], random_data)

    def _parallel_transport_bvp_is_isometry_data(
        self,
        metric_args_list,
        space_list,
        tangent_shape_list,
        n_tangent_vecs_list,
        is_tangent_atol=gs.atol,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        random_data = []
        for metric_args, space, tangent_shape, n_tangent_vecs in zip(
            metric_args_list, space_list, tangent_shape_list, n_tangent_vecs_list
        ):
            base_point = space.random_point()

            tangent_vec = space.to_tangent(
                gs.random.normal(size=(n_tangent_vecs,) + tangent_shape), base_point
            )
            end_point = space.random_point()
            random_data.append(
                dict(
                    metric_args=metric_args,
                    space=space,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    end_point=end_point,
                    is_tangent_atol=is_tangent_atol,
                    rtol=rtol,
                    atol=atol,
                )
            )

        return self.generate_tests([], random_data)
