import itertools

import pytest

import geomstats.backend as gs


def better_squeeze(array):
    """Delete possible singleton dimension on first axis."""
    if len(array) == 1:
        return gs.squeeze(array, axis=0)
    return array


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


class _ManifoldTestData(TestData):
    """Class for ManifoldTestData: data to test manifold properties."""

    def _random_point_belongs_test_data(
        self,
        smoke_space_args_list,
        smoke_n_points_list,
        space_args_list,
        n_points_list,
        belongs_atol=gs.atol,
    ):
        """Generate data to check that a random point belongs to the manifold.

        Parameters
        ----------
        smoke_space_args_list : list
            List of spaces' args on which smoke tests will run.
        smoke_n_points_list : list
            Integers representing the numbers of points on which smoke tests will run.
        space_args_list : list
            List of spaces' (manifolds') args on which randomized tests will run.
        n_points_list : list
            List of integers as numbers of points on which randomized tests will run.
        belongs_atol : float
            Absolute tolerance for the belongs function.
        """
        smoke_data = [
            dict(space_args=space_args, n_points=n_points, belongs_atol=belongs_atol)
            for space_args, n_points in zip(smoke_space_args_list, smoke_n_points_list)
        ]
        random_data = [
            dict(space_args=space_args, n_points=n_points, belongs_atol=belongs_atol)
            for space_args, n_points in zip(space_args_list, n_points_list)
        ]
        return self.generate_tests(smoke_data, random_data)

    def _projection_belongs_test_data(
        self, space_args_list, shape_list, n_samples_list, belongs_atol=gs.atol
    ):
        """Generate data to check that a point projected on a manifold belongs to the manifold.

        Parameters
        ----------
        space_args_list : list
            List of spaces' args on which tests will run.
        shape_list : list
            List of shapes of the random data that is generated, and projected.
        n_samples_list : list
            List of integers for the number of random data is generated, and projected.
        belongs_atol : float
            Absolute tolerance for the belongs function.
        """
        random_data = [
            dict(
                space_args=space_args,
                data=gs.random.normal(size=(n_samples,) + shape),
                belongs_atol=belongs_atol,
            )
            for space_args, shape, n_samples in zip(
                space_args_list, shape_list, n_samples_list
            )
        ]
        return self.generate_tests([], random_data)

    def _to_tangent_is_tangent_test_data(
        self,
        space_cls,
        space_args_list,
        shape_list,
        n_vecs_list,
        is_tangent_atol=gs.atol,
    ):
        """Generate data to check that to_tangent returns a tangent vector.

        Parameters
        ----------
        space_cls : Manifold
            Class of the space, i.e. a child class of Manifold.
        space_args_list : list
            List of spaces' args on which tests will run.
        shape_list : list
            List of shapes of the random vectors generated, and projected.
        n_vecs_list : list
            List of integers for the number of random vectors generated, and projected.
        is_tangent_atol : float
            Absolute tolerance for the is_tangent function.
        """
        random_data = []

        for space_args, shape, n_vecs in zip(space_args_list, shape_list, n_vecs_list):
            space = space_cls(*space_args)
            vec = gs.random.normal(size=(n_vecs,) + shape)
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


class _OpenSetTestData(_ManifoldTestData):
    def _to_tangent_is_tangent_in_ambient_space_test_data(
        self, space_cls, space_args_list, shape_list, is_tangent_atol=gs.atol
    ):
        """Generate data to check that tangent vectors are in ambient space's tangent space.

        Parameters
        ----------
        space_cls : Manifold
            Class of the space, i.e. a child class of Manifold.
        space_args_list : list
            Arguments to pass to constructor of the manifold.
        shape_list : list
            List of shapes of the random data that is generated, and projected.
        """
        random_data = [
            dict(
                space_args=space_args,
                vector=gs.random.normal(size=shape),
                base_point=space_cls(*space_args).random_point(shape[0]),
                is_tangent_atol=is_tangent_atol,
            )
            for space_args, shape in zip(space_args_list, shape_list)
        ]
        return self.generate_tests([], random_data)


class _LevelSetTestData(_ManifoldTestData):
    def _extrinsic_intrinsic_composition_test_data(
        self, space_cls, space_args_list, n_samples_list, rtol=gs.rtol, atol=gs.atol
    ):
        """Generate data to check that changing coordinate system twice gives back the point.

        Assumes that random_point generates points in extrinsic coordinates.

        Parameters
        ----------
        space_cls : Manifold
            Class of the space, i.e. a child class of Manifold.
        space_args_list : list
            Arguments to pass to constructor of the manifold.
        n_samples_list : list
            List of number of extrinsic points to generate.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = [
            dict(
                space_args=space_args,
                point_extrinsic=space_cls(
                    *space_args, default_coords_type="extrinsic"
                ).random_point(n_samples),
                rtol=rtol,
                atol=atol,
            )
            for space_args, n_samples in zip(space_args_list, n_samples_list)
        ]
        return self.generate_tests([], random_data)

    def _intrinsic_extrinsic_composition_test_data(
        self, space_cls, space_args_list, n_samples_list, rtol=gs.rtol, atol=gs.atol
    ):
        """Generate data to check that changing coordinate system twice gives back the point.

        Assumes that the first elements in space_args is the dimension of the space.

        Parameters
        ----------
        space_cls : Manifold
            Class of the space, i.e. a child class of Manifold.
        space_args_list : list
            Arguments to pass to constructor of the manifold.
        n_samples_list : list
            List of number of intrinsic points to generate.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for space_args, n_samples in zip(space_args_list, n_samples_list):

            space = space_cls(*space_args, default_coords_type="intrinsic")
            point_intrinic = space.random_point(n_samples)
            random_data.append(
                dict(
                    space_args=space_args,
                    point_intrinic=point_intrinic,
                    rtol=rtol,
                    atol=atol,
                )
            )
        return self.generate_tests([], random_data)


class _LieGroupTestData(_ManifoldTestData):
    def _exp_log_composition_test_data(
        self,
        group_cls,
        group_args_list,
        shape_list,
        n_samples_list,
        smoke_data=None,
        amplitude=1.0,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        """Generate data to check that group exponential and logarithm are inverse.

        Parameters
        ----------
        group_cls : LieGroup
            Class of the group, i.e. a child class of LieGroup.
        group_args_list : list
            Arguments to pass to constructor of the Lie group.
        n_samples_list : list
            List of number of points and tangent vectors to generate.
        """
        random_data = []
        for group_args, shape, n_samples in zip(
            group_args_list, shape_list, n_samples_list
        ):
            group = group_cls(*group_args)
            for base_point in [group.random_point(), group.identity]:
                tangent_vec = group.to_tangent(
                    gs.random.normal(size=(n_samples,) + shape) / amplitude, base_point
                )
                random_data.append(
                    dict(
                        group_args=group_args,
                        tangent_vec=tangent_vec,
                        base_point=base_point,
                        rtol=rtol,
                        atol=atol,
                    )
                )

        if smoke_data is None:
            smoke_data = []
        return self.generate_tests(smoke_data, random_data)

    def _log_exp_composition_test_data(
        self,
        group_cls,
        group_args_list,
        n_samples_list,
        smoke_data=None,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        """Generate data to check that group logarithm and exponential are inverse.

        Parameters
        ----------
        group_cls : LieGroup
            Class of the group, i.e. a child class of LieGroup.
        group_args_list : list
            List of arguments to pass to constructor of the Lie group.
        n_samples_list : list
            List of number of points and tangent vectors to generate.
        """
        random_data = []
        for group_args, n_samples in zip(group_args_list, n_samples_list):
            group = group_cls(*group_args)
            for base_point in [group.random_point(), group.identity]:
                point = group.random_point(n_samples)
                random_data.append(
                    dict(
                        group_args=group_args,
                        point=point,
                        base_point=base_point,
                        rtol=rtol,
                        atol=atol,
                    )
                )
        if smoke_data is None:
            smoke_data = []
        return self.generate_tests(smoke_data, random_data)


class _VectorSpaceTestData(_ManifoldTestData):
    def _basis_belongs_test_data(self, space_args_list, belongs_atol=gs.atol):
        """Generate data to check that basis elements belong to vector space.

        Parameters
        ----------
        space_args_list : list
            List of arguments to pass to constructor of the vector space.
        belongs_atol : float
            Absolute tolerance of the belongs function.
        """
        random_data = [
            dict(space_args=space_args, belongs_atol=belongs_atol)
            for space_args in space_args_list
        ]
        return self.generate_tests([], random_data)

    def _basis_cardinality_test_data(self, space_args_list):
        """Generate data to check that the number of basis elements is the dimension.

        Parameters
        ----------
        space_args_list : list
            List of arguments to pass to constructor of the vector space.
        """
        random_data = [dict(space_args=space_args) for space_args in space_args_list]
        return self.generate_tests([], random_data)


class _MatrixLieAlgebraTestData(_VectorSpaceTestData):
    def _basis_representation_matrix_representation_composition_test_data(
        self, space_cls, space_args_list, n_samples_list, rtol=gs.rtol, atol=gs.atol
    ):
        """Generate data to check that changing coordinates twice gives back the point.

        Parameters
        ----------
        space_cls : LieAlgebra
            Class of the space, i.e. a child class of LieAlgebra.
        space_args_list : list
            Arguments to pass to constructor of the manifold.
        n_samples_list : list
            List of numbers of samples to generate.
        """
        random_data = [
            dict(
                space_args=space_args,
                matrix_rep=space_cls(*space_args).random_point(n_samples),
                rtol=rtol,
                atol=atol,
            )
            for space_args, n_samples in zip(space_args_list, n_samples_list)
        ]
        return self.generate_tests([], random_data)

    def _matrix_representation_basis_representation_composition_test_data(
        self, space_cls, space_args_list, n_samples_list, rtol=gs.rtol, atol=gs.atol
    ):
        """Generate data to check that changing coordinates twice gives back the point.

        Parameters
        ----------
        space_cls : LieAlgebra
            Class of the space, i.e. a child class of LieAlgebra.
        space_args_list : list
            Arguments to pass to constructor of the LieAlgebra.
        n_samples_list : list
            List of numbers of samples to generate.
        """
        random_data = [
            dict(
                space_args=space_args,
                basis_rep=space_cls(*space_args).basis_representation(
                    space_cls(*space_args).random_point(n_samples)
                ),
                rtol=rtol,
                atol=atol,
            )
            for space_args, n_samples in zip(space_args_list, n_samples_list)
        ]
        return self.generate_tests([], random_data)


class _ConnectionTestData(TestData):
    def _exp_shape_test_data(
        self, connection_args_list, space_list, shape_list, n_samples_list
    ):
        """Generate data to check that exp returns an array of the expected shape.

        Parameters
        ----------
        connection_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        shape_list : list
            List of shapes for random data to generate.
        n_samples_list : list
            List of number of random data to generate.
        """
        random_data = []
        for connection_args, space, tangent_shape, n_samples in zip(
            connection_args_list, space_list, shape_list, n_samples_list
        ):
            base_point = space.random_point(n_samples)
            tangent_vec = space.to_tangent(
                gs.random.normal(size=(n_samples,) + tangent_shape), base_point
            )
            n_points_list = itertools.product([1, n_samples], [1, n_samples])
            expected_shape_list = [space.shape] + [(n_samples,) + space.shape] * 3
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

    def _log_shape_test_data(self, connection_args_list, space_list, n_samples_list):
        """Generate data to check that log returns an array of the expected shape.

        Parameters
        ----------
        connection_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        n_samples_list : list
            List of number of random data to generate.
        """
        random_data = []
        for connection_args, space, n_samples in zip(
            connection_args_list, space_list, n_samples_list
        ):
            base_point = space.random_point(n_samples)
            point = space.random_point(n_samples)
            n_points_list = itertools.product([1, n_samples], [1, n_samples])
            expected_shape_list = [space.shape] + [(n_samples,) + space.shape] * 3
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

    def _exp_belongs_test_data(
        self,
        connection_args_list,
        space_list,
        shape_list,
        n_samples_list,
        belongs_atol=gs.atol,
    ):
        """Generate data to check that exp gives a point on the manifold.

        Parameters
        ----------
        connection_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        shape_list : list
            List of shapes for random data to generate.
        n_samples_list : list
            List of number of random data to generate.
        """
        random_data = []
        for connection_args, space, shape, n_tangent_vecs in zip(
            connection_args_list, space_list, shape_list, n_samples_list
        ):
            base_point = space.random_point()
            tangent_vec = space.to_tangent(
                gs.random.normal(size=(n_tangent_vecs,) + shape), base_point
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

    def _log_is_tangent_test_data(
        self, connection_args_list, space_list, n_samples_list, is_tangent_atol=gs.atol
    ):
        """Generate data to check that log gives a tangent vector.

        Parameters
        ----------
        connection_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        n_samples_list : list
            List of number of random data to generate.
        """
        random_data = []
        for connection_args, space, n_samples in zip(
            connection_args_list, space_list, n_samples_list
        ):
            point = space.random_point(n_samples)
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

    def _geodesic_ivp_belongs_test_data(
        self,
        connection_args_list,
        space_list,
        shape_list,
        n_points_list,
        belongs_atol=gs.atol,
    ):
        """Generate data to check that connection geodesics belong to manifold.

        Parameters
        ----------
        connection_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        shape_list : list
            List of shapes for random data to generate.
        n_points_list : list
            List of number of times on the geodesics.
        belongs_atol : float
            Absolute tolerance for the belongs function.
        """
        random_data = []
        for connection_args, space, n_points, shape in zip(
            connection_args_list, space_list, n_points_list, shape_list
        ):
            initial_point = space.random_point()
            initial_tangent_vec = space.to_tangent(
                gs.random.normal(size=shape), initial_point
            )
            random_data.append(
                dict(
                    connection_args=connection_args,
                    space=space,
                    n_points=n_points,
                    initial_point=initial_point,
                    initial_tangent_vec=initial_tangent_vec,
                    belongs_atol=belongs_atol,
                )
            )
        return self.generate_tests([], random_data)

    def _geodesic_bvp_belongs_test_data(
        self,
        connection_args_list,
        space_list,
        n_points_list,
        belongs_atol=gs.atol,
    ):
        """Generate data to check that connection geodesics belong to manifold.

        Parameters
        ----------
        connection_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        n_points_list : list
            List of number of points on the geodesics.
        belongs_atol : float
            Absolute tolerance for the belongs function.
        """
        random_data = []
        for connection_args, space, n_points in zip(
            connection_args_list,
            space_list,
            n_points_list,
        ):
            initial_point = space.random_point()
            end_point = space.random_point()
            random_data.append(
                dict(
                    connection_args=connection_args,
                    space=space,
                    n_points=n_points,
                    initial_point=initial_point,
                    end_point=end_point,
                    belongs_atol=belongs_atol,
                )
            )
        return self.generate_tests([], random_data)

    def _log_exp_composition_test_data(
        self,
        connection_args_list,
        space_list,
        n_samples_list,
        smoke_data=None,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        """Generate data to check that logarithm and exponential are inverse.

        Parameters
        ----------
        connection_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        n_samples_list : list
            List of number of random data to generate.
        """
        random_data = []
        for connection_args, space, n_samples in zip(
            connection_args_list, space_list, n_samples_list
        ):
            point = space.random_point(n_samples)
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
        if smoke_data is None:
            smoke_data = []
        return self.generate_tests(smoke_data, random_data)

    def _exp_log_composition_test_data(
        self,
        connection_args_list,
        space_list,
        shape_list,
        n_samples_list,
        smoke_data=None,
        amplitude=1.0,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        """Generate data to check that exponential and logarithm are inverse.

        Parameters
        ----------
        connection_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        shape_list : list
            List of shapes for random data to generate.
        n_samples_list : list
            List of number of random data to generate.
        amplitude : float
            Factor to scale the amplitude of a tangent vector to stay in the
            injectivity domain of the exponential map.
        """
        random_data = []
        for connection_args, space, shape, n_samples in zip(
            connection_args_list, space_list, shape_list, n_samples_list
        ):
            base_point = space.random_point()
            tangent_vec = space.to_tangent(
                gs.random.normal(size=(n_samples,) + shape) / amplitude, base_point
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
        if smoke_data is None:
            smoke_data = []
        return self.generate_tests(smoke_data, random_data)

    def _exp_ladder_parallel_transport_test_data(
        self,
        connection_args_list,
        space_list,
        shape_list,
        n_samples_list,
        n_rungs_list,
        alpha_list,
        scheme_list,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        """Generate data to check that end point of ladder matches exponential.

        Parameters
        ----------
        connection_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        shape_list : list
            List of shapes for random data to generate.
        n_rungs_list : list
            List of number of rungs for the ladder.
        alpha_list : list
            List of exponents for th scaling of the vector to transport.
        scheme_list : list
            List of ladder schemes to test.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for (connection_args, space, shape, n_samples, n_rungs, alpha, scheme,) in zip(
            connection_args_list,
            space_list,
            shape_list,
            n_samples_list,
            n_rungs_list,
            alpha_list,
            scheme_list,
        ):
            base_point = space.random_point()

            tangent_vec = space.to_tangent(
                gs.random.normal(size=(n_samples,) + shape), base_point
            )
            direction = space.to_tangent(gs.random.normal(size=shape), base_point)
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

    def _exp_geodesic_ivp_test_data(
        self,
        connection_args_list,
        space_list,
        shape_list,
        n_samples_list,
        n_points_list,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        """Generate data to check that end point of geodesic matches exponential.

        Parameters
        ----------
        connection_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        shape_list : list
            List of shapes for random data to generate.
        n_samples_list : list
            List of number of random data to generate.
        n_points_list : list
            List of number of times on the geodesics.
        belongs_atol : float
            Absolute tolerance for the belongs function.
        """
        random_data = []
        for connection_args, space, shape, n_samples, n_points in zip(
            connection_args_list,
            space_list,
            shape_list,
            n_samples_list,
            n_points_list,
        ):
            base_point = space.random_point()
            tangent_vec = space.to_tangent(
                gs.random.normal(size=(n_samples,) + shape), base_point
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


class _RiemannianMetricTestData(_ConnectionTestData):
    def _squared_dist_is_symmetric_test_data(
        self,
        metric_args_list,
        space_list,
        n_points_a_list,
        n_points_b_list,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        """Generate data to check that the squared geodesic distance is symmetric.

        Parameters
        ----------
        metric_args_list : list
            List of arguments to pass to constructor of the metric.
        space_list : list
            List of spaces on which the metric is defined.
        n_points_a_list : list
            List of number of points A to generate on the manifold.
        n_points_b_list : list
            List of number of points B to generate on the manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for metric_args, space, n_points_a, n_points_b in zip(
            metric_args_list, space_list, n_points_a_list, n_points_b_list
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

    def _parallel_transport_ivp_is_isometry_test_data(
        self,
        metric_args_list,
        space_list,
        shape_list,
        n_samples_list,
        is_tangent_atol=gs.atol,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        """Generate data to check that parallel transport is an isometry.

        Parameters
        ----------
        metric_args_list : list
            List of arguments to pass to constructor of the metric.
        space_list : list
            List of spaces on which the metric is defined.
        shape_list : list
            List of shapes for random data to generate.
        n_samples_list : list
            List of number of random data to generate.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for metric_args, space, shape, n_samples in zip(
            metric_args_list, space_list, shape_list, n_samples_list
        ):
            base_point = space.random_point()

            tangent_vec = space.to_tangent(
                gs.random.normal(size=(n_samples,) + shape), base_point
            )
            direction = space.to_tangent(gs.random.normal(size=shape), base_point)
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

    def _parallel_transport_bvp_is_isometry_test_data(
        self,
        metric_args_list,
        space_list,
        shape_list,
        n_samples_list,
        is_tangent_atol=gs.atol,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        """Generate data to check that parallel transport is an isometry.

        Parameters
        ----------
        metric_args_list : list
            List of arguments to pass to constructor of the metric.
        space_list : list
            List of spaces on which the metric is defined.
        shape_list : list
            List of shapes for random data to generate.
        n_samples_list : list
            List of number of random data to generate.
        is_tangent_atol: float
            Asbolute tolerance for the is_tangent function.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for metric_args, space, tangent_shape, n_tangent_vecs in zip(
            metric_args_list, space_list, shape_list, n_samples_list
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
