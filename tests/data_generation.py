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
        for test_data, marker in zip(
            [smoke_test_data, random_test_data], [pytest.mark.smoke, pytest.mark.random]
        ):
            for test_datum in test_data:
                if isinstance(test_datum, dict):
                    test_datum["marks"] = marker
                else:
                    test_datum = list(test_datum)
                    test_datum.append(marker)

                tests.append(test_datum)

        return tests


class _ManifoldTestData(TestData):
    """Class for ManifoldTestData: data to test manifold properties."""

    def random_point_belongs_test_data(
        self,
        belongs_atol=gs.atol,
    ):
        """Generate data to check that a random point belongs to the manifold.

        Parameters
        ----------
        belongs_atol : float
            Absolute tolerance for the belongs function.
        """
        random_data = [
            dict(space_args=space_args, n_points=n_points, belongs_atol=belongs_atol)
            for space_args, n_points in zip(self.space_args_list, self.n_points_list)
        ]
        return self.generate_tests([], random_data)

    def projection_belongs_test_data(self, belongs_atol=gs.atol):
        """Generate data to check that a point projected on a manifold belongs to the manifold.

        Parameters
        ----------
        belongs_atol : float
            Absolute tolerance for the belongs function.
        """
        random_data = [
            dict(
                space_args=space_args,
                point=gs.random.normal(size=(n_points,) + shape),
                belongs_atol=belongs_atol,
            )
            for space_args, shape, n_points in zip(
                self.space_args_list, self.shape_list, self.n_points_list
            )
        ]
        return self.generate_tests([], random_data)

    def to_tangent_is_tangent_test_data(
        self,
        is_tangent_atol=gs.atol,
    ):
        """Generate data to check that to_tangent returns a tangent vector.

        Parameters
        ----------
        is_tangent_atol : float
            Absolute tolerance for the is_tangent function.
        """
        random_data = []

        for space_args, shape, n_vecs in zip(
            self.space_args_list, self.shape_list, self.n_vecs_list
        ):
            space = self.Space(*space_args)
            vec = gs.random.normal(size=(n_vecs,) + shape)
            base_point = space.random_point()
            random_data.append(
                dict(
                    space_args=space_args,
                    vector=vec,
                    base_point=base_point,
                    is_tangent_atol=is_tangent_atol,
                )
            )
        return self.generate_tests([], random_data)

    def random_tangent_vec_is_tangent_test_data(
        self,
        is_tangent_atol=gs.atol,
    ):
        """Generate data to check that random tangent vec returns a tangent vector.

        Parameters
        ----------
        is_tangent_atol : float
            Absolute tolerance for the is_tangent function.
        """
        random_data = []

        # TODO: n_vecs_list or self.n_tangent_vecs_list?

        for space_args, n_tangent_vec in zip(self.space_args_list, self.n_vecs_list):
            space = self.Space(*space_args)
            base_point = space.random_point()
            random_data.append(
                dict(
                    space_args=space_args,
                    n_samples=n_tangent_vec,
                    base_point=base_point,
                    is_tangent_atol=is_tangent_atol,
                )
            )
        return self.generate_tests([], random_data)


class _OpenSetTestData(_ManifoldTestData):
    def to_tangent_is_tangent_in_ambient_space_test_data(self):
        """Generate data to check that tangent vectors are in ambient space's
        tangent space.
        """
        random_data = [
            dict(
                space_args=space_args,
                vector=gs.random.normal(size=shape),
                base_point=self.Space(*space_args).random_point(shape[0]),
            )
            for space_args, shape in zip(self.space_args_list, self.shape_list)
        ]
        return self.generate_tests([], random_data)


class _LevelSetTestData(_ManifoldTestData):
    def intrinsic_after_extrinsic_test_data(self):
        """Generate data to check that changing coordinate system twice gives back the point.

        Assumes that random_point generates points in extrinsic coordinates.
        """
        random_data = [
            dict(
                space_args=space_args,
                point_extrinsic=self.Space(
                    *space_args, default_coords_type="extrinsic"
                ).random_point(n_points),
            )
            for space_args, n_points in zip(self.space_args_list, self.n_points_list)
        ]
        return self.generate_tests([], random_data)

    def extrinsic_after_intrinsic_test_data(self):
        """Generate data to check that changing coordinate system twice gives back the point.

        Assumes that the first elements in space_args is the dimension of the space.
        """
        random_data = []
        for space_args, n_points in zip(self.space_args_list, self.n_points_list):

            space = self.Space(*space_args, default_coords_type="intrinsic")
            point_intrinsic = space.random_point(n_points)
            random_data.append(
                dict(
                    space_args=space_args,
                    point_intrinsic=point_intrinsic,
                )
            )
        return self.generate_tests([], random_data)


class _LieGroupTestData(_ManifoldTestData):
    def _generate_compose_data(self):
        random_data = []
        for group_args, n_points in zip(self.space_args_list, self.n_points_list):

            group = self.Space(*group_args)
            point = group.random_point(n_points)
            random_data.append(dict(group_args=group_args, point=point))

        return self.generate_tests([], random_data)

    def compose_point_with_inverse_point_is_identity_test_data(self):
        """Generate data to check composition of point, inverse is identity."""
        return self._generate_compose_data()

    def compose_inverse_point_with_point_is_identity_test_data(self):
        """Generate data to check composition of inverse, point is identity."""
        return self._generate_compose_data()

    def compose_point_with_identity_is_point_test_data(self):
        """Generate data to check composition of point, identity is point."""
        return self._generate_compose_data()

    def compose_identity_with_point_is_point_test_data(self):
        """Generate data to check composition of identity, point is point."""
        return self._generate_compose_data()

    def log_after_exp_test_data(self, amplitude=1.0):
        """Generate data to check that group exponential and logarithm are inverse."""
        random_data = []
        for group_args, shape, n_tangent_vecs in zip(
            self.space_args_list, self.shape_list, self.n_tangent_vecs_list
        ):
            group = self.Space(*group_args)
            for base_point in [group.random_point(), group.identity]:

                tangent_vec = group.to_tangent(
                    gs.random.normal(size=(n_tangent_vecs,) + shape) / amplitude,
                    base_point,
                )
                random_data.append(
                    dict(
                        group_args=group_args,
                        tangent_vec=tangent_vec,
                        base_point=base_point,
                    )
                )

        return self.generate_tests([], random_data)

    def exp_after_log_test_data(self):
        """Generate data to check that group logarithm and exponential are inverse."""
        random_data = []
        for group_args, n_points in zip(self.space_args_list, self.n_points_list):
            group = self.Space(*group_args)
            for base_point in [group.random_point(), group.identity]:
                point = group.random_point(n_points)
                random_data.append(
                    dict(
                        group_args=group_args,
                        point=point,
                        base_point=base_point,
                    )
                )
        return self.generate_tests([], random_data)

    def to_tangent_at_identity_belongs_to_lie_algebra_test_data(self):
        """Generate data to check that to tangent at identity belongs to lie algebra."""
        random_data = []

        for group_args, shape, n_vecs in zip(
            self.space_args_list, self.shape_list, self.n_vecs_list
        ):
            vec = gs.random.normal(size=(n_vecs,) + shape)
            random_data.append(dict(group_args=group_args, vector=vec))
        return self.generate_tests([], random_data)


class _VectorSpaceTestData(_ManifoldTestData):
    def basis_belongs_test_data(self):
        """Generate data to check that basis elements belong to vector space."""
        random_data = [
            dict(space_args=space_args) for space_args in self.space_args_list
        ]

        return self.generate_tests([], random_data)

    def basis_cardinality_test_data(self):
        """Generate data to check that the number of basis elements is the dimension."""
        random_data = [
            dict(space_args=space_args) for space_args in self.space_args_list
        ]
        return self.generate_tests([], random_data)

    def random_point_is_tangent_test_data(self, is_tangent_atol=gs.atol):
        """Generate data to check that random point is tangent vector.

        Parameters
        ----------
        is_tangent_atol : float
            Absolute tolerance for the is_tangent function.
        """
        random_data = []
        for space_args, n_points in zip(self.space_args_list, self.n_points_list):
            random_data += [
                dict(
                    space_args=space_args,
                    n_points=n_points,
                    is_tangent_atol=is_tangent_atol,
                )
            ]

        return self.generate_tests([], random_data)

    def to_tangent_is_projection_test_data(
        self,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        """Generate data to check that to_tangent return projection.

        Parameters
        ----------
        space_cls : Manifold
            Class of the space, i.e. a child class of Manifold.
        space_args_list : list
            List of spaces' args on which tests will run.
        shape_list : list
            List of shapes of the random vectors generated, and projected.
        n_vecs_list : list
            List of integers for the number of random vectors generated.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for space_args, shape, n_vecs in zip(
            self.space_args_list, self.shape_list, self.n_vecs_list
        ):
            space = self.Space(*space_args)
            vec = gs.random.normal(size=(n_vecs,) + shape)
            base_point = space.random_point()

            random_data.append(
                dict(
                    space_args=space_args,
                    vector=vec,
                    base_point=base_point,
                    rtol=rtol,
                    atol=atol,
                )
            )

        return self.generate_tests([], random_data)


class _MatrixLieAlgebraTestData(_VectorSpaceTestData):
    def matrix_representation_after_basis_representation_test_data(self):
        """Generate data to check that changing coordinates twice gives back
        the point.
        """
        random_data = [
            dict(
                algebra_args=space_args,
                matrix_rep=self.Space(*space_args).random_point(n_points),
            )
            for space_args, n_points in zip(self.space_args_list, self.n_points_list)
        ]
        return self.generate_tests([], random_data)

    def basis_representation_after_matrix_representation_test_data(self):
        """Generate data to check that changing coordinates twice gives back
        the point.
        """
        random_data = [
            dict(
                algebra_args=space_args,
                basis_rep=self.Space(*space_args).basis_representation(
                    self.Space(*space_args).random_point(n_points)
                ),
            )
            for space_args, n_points in zip(self.space_args_list, self.n_points_list)
        ]
        return self.generate_tests([], random_data)


class _FiberBundleTestData(TestData):
    def is_horizontal_after_horizontal_projection_test_data(self):
        random_data = []
        for space_args, n_points in zip(self.space_args_list, self.n_points_list):
            space = self.Space(*space_args)
            base_point = space.random_point(n_points)
            tangent_vec = space.random_tangent_vec(base_point, n_points)
            data = dict(
                space_args=space_args,
                base_point=base_point,
                tangent_vec=tangent_vec,
            )
            random_data.append(data)
        return self.generate_tests([], random_data)

    def is_vertical_after_vertical_projection_test_data(self):
        random_data = []
        for space_args, n_points in zip(self.space_args_list, self.n_points_list):
            space = self.Space(*space_args)
            base_point = space.random_point(n_points)
            tangent_vec = space.random_tangent_vec(base_point, n_points)
            data = dict(
                space_args=space_args,
                base_point=base_point,
                tangent_vec=tangent_vec,
            )
            random_data.append(data)

        return self.generate_tests([], random_data)

    def is_horizontal_after_log_after_align_test_data(self):
        random_data = [
            dict(
                space_args=space_args,
                base_point=self.Space(*space_args).random_point(n_base_points),
                point=self.Space(*space_args).random_point(n_points),
            )
            for space_args, n_points, n_base_points in zip(
                self.space_args_list, self.n_points_list, self.n_base_points_list
            )
        ]
        return self.generate_tests([], random_data)

    def riemannian_submersion_after_lift_test_data(self):
        random_data = [
            dict(
                space_args=space_args,
                base_point=self.Base(*space_args).random_point(n_points),
            )
            for space_args, n_points in zip(
                self.space_args_list, self.n_base_points_list
            )
        ]
        return self.generate_tests([], random_data)

    def is_tangent_after_tangent_riemannian_submersion_test_data(self):
        random_data = []
        for space_args, n_vecs in zip(self.space_args_list, self.n_vecs_list):
            base_point = self.Space(*space_args).random_point()
            tangent_vec = self.Space(*space_args).random_tangent_vec(base_point, n_vecs)
            d = dict(
                space_args=space_args,
                base_cls=self.Base,
                tangent_vec=tangent_vec,
                base_point=base_point,
            )
            random_data.append(d)
        return self.generate_tests([], random_data)


class _ConnectionTestData(TestData):
    def exp_shape_test_data(self):
        """Generate data to check that exp returns an array of the expected shape."""
        n_samples_list = [3] * len(self.metric_args_list)
        random_data = []
        for connection_args, space, tangent_shape, n_samples in zip(
            self.metric_args_list, self.space_list, self.shape_list, n_samples_list
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
                        expected=expected_shape,
                    )
                )
        return self.generate_tests([], random_data)

    def log_shape_test_data(self):
        """Generate data to check that log returns an array of the expected shape."""
        n_samples_list = [3] * len(self.metric_args_list)
        random_data = []
        for connection_args, space, n_samples in zip(
            self.metric_args_list, self.space_list, n_samples_list
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
                        expected=expected_shape,
                    )
                )
        return self.generate_tests([], random_data)

    def exp_belongs_test_data(self):
        """Generate data to check that exp gives a point on the manifold."""
        random_data = []
        for connection_args, space, shape, n_tangent_vecs in zip(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
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
                )
            )
        return self.generate_tests([], random_data)

    def log_is_tangent_test_data(self):
        """Generate data to check that log gives a tangent vector.

        Parameters
        ----------
        self.metric_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        n_samples_list : list
            List of number of random data to generate.
        """
        random_data = []
        for connection_args, space, n_points in zip(
            self.metric_args_list, self.space_list, self.n_points_list
        ):
            point = space.random_point(n_points)
            base_point = space.random_point()
            random_data.append(
                dict(
                    connection_args=connection_args,
                    space=space,
                    point=point,
                    base_point=base_point,
                )
            )
        return self.generate_tests([], random_data)

    def geodesic_ivp_belongs_test_data(self):
        """Generate data to check that connection geodesics belong to manifold."""
        random_data = []
        for connection_args, space, n_points, shape in zip(
            self.metric_args_list, self.space_list, self.n_points_list, self.shape_list
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
                )
            )
        return self.generate_tests([], random_data)

    def geodesic_bvp_belongs_test_data(self):
        """Generate data to check that connection geodesics belong to manifold."""
        random_data = []
        for connection_args, space, n_points in zip(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
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
                )
            )
        return self.generate_tests([], random_data)

    def exp_after_log_test_data(self):
        """Generate data to check that logarithm and exponential are inverse."""
        random_data = []
        for connection_args, space, n_points in zip(
            self.metric_args_list, self.space_list, self.n_points_list
        ):
            point = space.random_point(n_points)
            base_point = space.random_point()
            random_data.append(
                dict(
                    connection_args=connection_args,
                    point=point,
                    base_point=base_point,
                )
            )
        return self.generate_tests([], random_data)

    def log_after_exp_test_data(
        self,
        amplitude=1.0,
    ):
        """Generate data to check that exponential and logarithm are inverse.

        Parameters
        ----------
        amplitude : float
            Factor to scale the amplitude of a tangent vector to stay in the
            injectivity domain of the exponential map.
        """
        random_data = []
        for connection_args, space, shape, n_tangent_vecs in zip(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        ):
            base_point = space.random_point()
            tangent_vec = space.to_tangent(
                gs.random.normal(size=(n_tangent_vecs,) + shape) / amplitude, base_point
            )
            random_data.append(
                dict(
                    connection_args=connection_args,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                )
            )
        return self.generate_tests([], random_data)

    def exp_ladder_parallel_transport_test_data(self):
        """Generate data to check that end point of ladder matches exponential."""
        random_data = []
        for (
            connection_args,
            space,
            shape,
            n_tangent_vecs,
            n_rungs,
            alpha,
            scheme,
        ) in zip(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            self.n_rungs_list,
            self.alpha_list,
            self.scheme_list,
        ):
            base_point = space.random_point()

            tangent_vec = space.to_tangent(
                gs.random.normal(size=(n_tangent_vecs,) + shape), base_point
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
                )
            )

        return self.generate_tests([], random_data)

    def exp_geodesic_ivp_test_data(self):
        """Generate data to check that end point of geodesic matches exponential."""
        random_data = []
        for connection_args, space, shape, n_tangent_vecs, n_points in zip(
            self.connection_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            self.n_points_list,
        ):
            base_point = space.random_point()
            tangent_vec = gs.squeeze(
                space.to_tangent(
                    gs.random.normal(size=(n_tangent_vecs,) + shape), base_point
                )
            )
            random_data.append(
                dict(
                    connection_args=connection_args,
                    n_points=n_points,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                )
            )
        return self.generate_tests([], random_data)

    def riemann_tensor_shape_test_data(self):
        """Generate data to check that riemann_tensor returns an array of the expected shape.

        Parameters
        ----------
        connection_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        shape_list : list
            List of shapes for random data to generate.
        """
        random_data = []
        for n_points, connection_args, space in zip(
            self.n_points_list,
            self.connection_args_list,
            self.space_list,
        ):
            base_point = space.random_point(n_points)
            expected_shape = (
                (n_points,) + space.shape * 4 if n_points >= 2 else space.shape * 4
            )
            random_data.append(
                dict(
                    connection_args=connection_args,
                    base_point=better_squeeze(base_point),
                    expected=expected_shape,
                )
            )
        return self.generate_tests([], random_data)

    def ricci_tensor_shape_test_data(self):
        """Generate data to check that ricci_tensor returns an array of the expected shape.

        Parameters
        ----------
        connection_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        shape_list : list
            List of shapes for random data to generate.
        """
        random_data = []
        for n_points, connection_args, space in zip(
            self.n_points_list,
            self.connection_args_list,
            self.space_list,
        ):
            base_point = space.random_point(n_points)
            expected_shape = (
                (n_points,) + space.shape * 2 if n_points >= 2 else space.shape * 2
            )
            random_data.append(
                dict(
                    connection_args=connection_args,
                    base_point=better_squeeze(base_point),
                    expected=expected_shape,
                )
            )
        return self.generate_tests([], random_data)

    def scalar_curvature_shape_test_data(self):
        """Generate data to check that scalar_curvature returns an array of the expected shape.

        Parameters
        ----------
        connection_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        shape_list : list
            List of shapes for random data to generate.
        """
        random_data = []
        for n_points, connection_args, space in zip(
            self.n_points_list,
            self.connection_args_list,
            self.space_list,
        ):
            base_point = space.random_point(n_points)
            expected_shape = expected_shape = (n_points,) if n_points >= 2 else ()
            random_data.append(
                dict(
                    connection_args=connection_args,
                    base_point=better_squeeze(base_point),
                    expected=expected_shape,
                )
            )
        return self.generate_tests([], random_data)


class _RiemannianMetricTestData(_ConnectionTestData):
    def dist_is_symmetric_test_data(self):
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
        return self.squared_dist_is_symmetric_test_data()

    def dist_is_positive_test_data(self):
        """Generate data to check that the squared geodesic distance is symmetric."""
        return self.squared_dist_is_positive_test_data()

    def squared_dist_is_symmetric_test_data(self):
        """Generate data to check that the squared geodesic distance is symmetric."""
        random_data = []
        for metric_args, space, n_points_a, n_points_b in zip(
            self.metric_args_list,
            self.space_list,
            self.n_points_a_list,
            self.n_points_b_list,
        ):
            point_a = space.random_point(n_points_a)
            point_b = space.random_point(n_points_b)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    point_a=point_a,
                    point_b=point_b,
                )
            )
        return self.generate_tests([], random_data)

    def squared_dist_is_positive_test_data(self):
        """Generate data to check that the squared geodesic distance is symmetric."""
        random_data = []
        for metric_args, space, n_points_a, n_points_b in zip(
            self.metric_args_list,
            self.space_list,
            self.n_points_a_list,
            self.n_points_b_list,
        ):
            point_a = space.random_point(n_points_a)
            point_b = space.random_point(n_points_b)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    point_a=point_a,
                    point_b=point_b,
                )
            )
        return self.generate_tests([], random_data)

    def dist_is_norm_of_log_test_data(self):
        """Generate data to check that the squared geodesic distance is symmetric."""
        return self.squared_dist_is_symmetric_test_data()

    def inner_product_is_symmetric_test_data(self):
        """Generate data to check that the inner product is symmetric."""
        random_data = []
        for metric_args, space, shape, n_tangent_vecs in zip(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        ):
            base_point = space.random_point()
            tangent_vec_a = space.to_tangent(
                gs.random.normal(size=(n_tangent_vecs,) + shape), base_point
            )
            tangent_vec_b = space.to_tangent(
                gs.random.normal(size=(n_tangent_vecs,) + shape), base_point
            )
            random_data.append(
                dict(
                    metric_args=metric_args,
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                )
            )
        return self.generate_tests([], random_data)

    def dist_point_to_itself_is_zero_test_data(self):
        """Generate data to check that the squared geodesic distance is symmetric."""
        random_data = []
        for metric_args, space, n_points in zip(
            self.metric_args_list, self.space_list, self.n_points_list
        ):
            point = space.random_point(n_points)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    point=point,
                )
            )
        return self.generate_tests([], random_data)

    def parallel_transport_ivp_is_isometry_test_data(self):
        """Generate data to check that parallel transport is an isometry."""
        random_data = []
        for metric_args, space, shape, n_tangent_vecs in zip(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        ):
            base_point = space.random_point()

            tangent_vec = space.to_tangent(
                gs.random.normal(size=(n_tangent_vecs,) + shape), base_point
            )
            direction = space.to_tangent(gs.random.normal(size=shape), base_point)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    space=space,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    direction=direction,
                )
            )

        return self.generate_tests([], random_data)

    def parallel_transport_bvp_is_isometry_test_data(self):
        """Generate data to check that parallel transport is an isometry."""
        random_data = []
        for metric_args, space, tangent_shape, n_tangent_vecs in zip(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
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
                )
            )

        return self.generate_tests([], random_data)

    def triangle_inequality_of_dist_test_data(self):
        """Generate data to check the traingle inequality of geodesic distance."""
        random_data = []
        for metric_args, space, n_points in zip(
            self.metric_args_list, self.space_list, self.n_points_list
        ):
            point_a = space.random_point(n_points)
            point_b = space.random_point(n_points)
            point_c = space.random_point(n_points)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    point_a=point_a,
                    point_b=point_b,
                    point_c=point_c,
                )
            )
        return self.generate_tests([], random_data)

    def covariant_riemann_tensor_is_skew_symmetric_1_test_data(self):
        """Generate data to check the first skew symmetry of covariant riemann tensor.

        Parameters
        ----------
        metric_args_list : list
            List of arguments to pass to constructor of the metric.
        space_list : list
            List of spaces on which the metric is defined.
        n_points_list : list
            List of number of random points to generate.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for metric_args, space, n_points in zip(
            self.metric_args_list, self.space_list, self.n_points_list
        ):
            base_point = space.random_point(n_points)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    base_point=base_point,
                )
            )
        return self.generate_tests([], random_data)

    def covariant_riemann_tensor_is_skew_symmetric_2_test_data(self):
        """Generate data to check the second skew symmetry of covariant riemann tensor.

        Parameters
        ----------
        metric_args_list : list
            List of arguments to pass to constructor of the metric.
        space_list : list
            List of spaces on which the metric is defined.
        n_points_list : list
            List of number of random points to generate.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for metric_args, space, n_points in zip(
            self.metric_args_list, self.space_list, self.n_points_list
        ):
            base_point = space.random_point(n_points)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    base_point=base_point,
                )
            )
        return self.generate_tests([], random_data)

    def covariant_riemann_tensor_bianchi_identity_test_data(self):
        """Generate data to check the bianchi identity of covariant riemann tensor.

        Parameters
        ----------
        metric_args_list : list
            List of arguments to pass to constructor of the metric.
        space_list : list
            List of spaces on which the metric is defined.
        n_points_list : list
            List of number of random points to generate.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for metric_args, space, n_points in zip(
            self.metric_args_list, self.space_list, self.n_points_list
        ):
            base_point = space.random_point(n_points)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    base_point=base_point,
                )
            )
        return self.generate_tests([], random_data)

    def covariant_riemann_tensor_is_interchange_symmetric_test_data(self):
        """Generate data to check the interchange symmetry of covariant riemann tensor.

        Parameters
        ----------
        metric_args_list : list
            List of arguments to pass to constructor of the metric.
        space_list : list
            List of spaces on which the metric is defined.
        n_points_list : list
            List of number of random points to generate.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for metric_args, space, n_points in zip(
            self.metric_args_list, self.space_list, self.n_points_list
        ):
            base_point = space.random_point(n_points)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    base_point=base_point,
                )
            )
        return self.generate_tests([], random_data)

    def sectional_curvature_shape_test_data(self):
        """Generate data to check that sectional_curvature returns an array of expected shape.

        Parameters
        ----------
        metric_args_list : list
            List of argument to pass to constructor of the metric.
        space_list : list
            List of spaces on which the metric is defined.
        shape_list : list
            List of shapes for random data to generate.
        """
        random_data = []
        for metric_args, n_points, space, shape, n_tangent_vecs in zip(
            self.metric_args_list,
            self.n_points_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        ):
            base_point = space.random_point(n_points)
            size = (n_tangent_vecs,) + shape if n_points == 1 else (n_points,) + shape
            tangent_vec_a = gs.squeeze(
                space.to_tangent(
                    gs.random.normal(size=size),
                    base_point,
                )
            )
            tangent_vec_b = gs.squeeze(
                space.to_tangent(
                    gs.random.normal(size=size),
                    base_point,
                )
            )
            random_data.append(
                dict(
                    metric_args=metric_args,
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=(n_points,) if n_points >= 2 else (n_tangent_vecs,),
                )
            )
        return self.generate_tests([], random_data)


class _InvariantMetricTestData(_RiemannianMetricTestData):
    def exp_at_identity_of_lie_algebra_belongs_test_data(self):
        random_data = []
        for metric_args, group, n_tangent_vecs in zip(
            self.metric_args_list, self.group_list, self.n_tangent_vecs_list
        ):
            lie_algebra_point = group.lie_algebra.random_point(n_tangent_vecs)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    group=group,
                    lie_algebra_point=lie_algebra_point,
                )
            )
        return self.generate_tests([], random_data)

    def log_at_identity_belongs_to_lie_algebra_test_data(self):
        random_data = []
        for metric_args, group, n_points in zip(
            self.metric_args_list, self.group_list, self.n_points_list
        ):
            point = group.random_point(n_points)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    group=group,
                    point=point,
                )
            )

        return self.generate_tests([], random_data)

    def exp_after_log_at_identity_test_data(self):
        random_data = []
        for metric_args, group, n_points in zip(
            self.metric_args_list, self.group_list, self.n_points_list
        ):
            point = group.random_point(n_points)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    group=group,
                    point=point,
                )
            )

        return self.generate_tests([], random_data)

    def log_after_exp_at_identity_test_data(
        self,
        amplitude=1.0,
    ):
        random_data = []

        for metric_args, group, shape, n_tangent_vecs in zip(
            self.metric_args_list,
            self.group_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        ):
            base_point = group.random_point()
            tangent_vec = group.to_tangent(
                gs.random.normal(size=(n_tangent_vecs,) + shape) / amplitude,
                base_point,
            )
            random_data.append(
                (
                    dict(
                        metric_args=metric_args,
                        group=group,
                        tangent_vec=tangent_vec,
                    )
                )
            )

        return self.generate_tests([], random_data)


class _QuotientMetricTestData(_RiemannianMetricTestData):
    def dist_is_smaller_than_bundle_dist_test_data(self):
        random_data = []
        for metric_args, bundle, n_points in zip(
            self.metric_args_list, self.bundle_list, self.n_points_list
        ):
            point_a = bundle.random_point(n_points)
            point_b = bundle.random_point(n_points)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    bundle=bundle,
                    point_a=point_a,
                    point_b=point_b,
                )
            )
        return self.generate_tests([], random_data)

    def log_is_horizontal_test_data(self):
        random_data = []
        for metric_args, bundle, n_points in zip(
            self.metric_args_list, self.bundle_list, self.n_points_list
        ):
            point = bundle.random_point(n_points)
            base_point = bundle.random_point()
            random_data.append(
                dict(
                    metric_args=metric_args,
                    bundle=bundle,
                    point=point,
                    base_point=base_point,
                )
            )

        return self.generate_tests([], random_data)


class _PointSetTestData(TestData):
    def random_point_belongs_test_data(self):

        random_data = [
            dict(space_args=space_args, n_points=n_points)
            for space_args, n_points in zip(self.space_args_list, self.n_points_list)
        ]

        return self.generate_tests([], random_data)

    def random_point_output_shape_test_data(self):
        space = self._PointSet(*self.space_args_list[0])

        smoke_data = [
            dict(space=space, n_samples=1),
            dict(space=space, n_samples=2),
        ]

        return self.generate_tests(smoke_data)

    def set_to_array_output_shape_test_data(self):
        space = self._PointSet(*self.space_args_list[0])
        pts = space.random_point(2)

        smoke_data = [
            dict(space=space, points=pts[0]),
            dict(space=space, points=pts),
        ]

        return self.generate_tests(smoke_data)


class _PointTestData(TestData):
    pass


class _PointMetricTestData(TestData):
    def dist_output_shape_test_data(self):
        space = self._PointSet(*self.space_args_list[0])
        metric = self._PointSetMetric(space)
        pts = space.random_point(2)

        dist_fnc = metric.dist

        smoke_data = [
            dict(dist_fnc=dist_fnc, point_a=pts[0], point_b=pts[1]),
            dict(dist_fnc=dist_fnc, point_a=pts[0], point_b=pts),
            dict(dist_fnc=dist_fnc, point_a=pts, point_b=pts[0]),
            dict(dist_fnc=dist_fnc, point_a=pts, point_b=pts),
        ]

        return self.generate_tests(smoke_data)

    def dist_properties_test_data(self):
        space = self._PointSet(*self.space_args_list[0])
        metric = self._PointSetMetric(space)
        pts = space.random_point(3)

        dist_fnc = metric.dist

        smoke_data = [
            dict(dist_fnc=dist_fnc, point_a=pts[0], point_b=pts[1], point_c=pts[2]),
        ]

        return self.generate_tests(smoke_data)

    def geodesic_output_shape_test_data(self):
        space = self._PointSet(*self.space_args_list[0])
        metric = self._PointSetMetric(space)
        pts = space.random_point(2)

        smoke_data = [
            dict(metric=metric, start_point=pts[0], end_point=pts[0], t=0.0),
            dict(metric=metric, start_point=pts[0], end_point=pts[0], t=[0.0, 1.0]),
            dict(metric=metric, start_point=pts[0], end_point=pts, t=0.0),
            dict(metric=metric, start_point=pts[0], end_point=pts, t=[0.0, 1.0]),
            dict(metric=metric, start_point=pts, end_point=pts[0], t=0.0),
            dict(metric=metric, start_point=pts, end_point=pts[0], t=[0.0, 1.0]),
            dict(metric=metric, start_point=pts, end_point=pts, t=0.0),
            dict(metric=metric, start_point=pts, end_point=pts, t=[0.0, 1.0]),
        ]

        return self.generate_tests(smoke_data)

    def geodesic_bounds_test_data(self):
        space = self._PointSet(*self.space_args_list[0])
        metric = self._PointSetMetric(space)
        pts = space.random_point(2)

        smoke_data = [dict(metric=metric, start_point=pts[0], end_point=pts[1])]

        return self.generate_tests(smoke_data)
