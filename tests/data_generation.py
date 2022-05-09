import itertools
import random

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
            space = self.space(*space_args)
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
            space = self.space(*space_args)
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
    def _intrinsic_after_extrinsic_test_data(
        self, space_cls, space_args_list, n_points_list, rtol=gs.rtol, atol=gs.atol
    ):
        """Generate data to check that changing coordinate system twice gives back the point.

        Assumes that random_point generates points in extrinsic coordinates.

        Parameters
        ----------
        space_cls : Manifold
            Class of the space, i.e. a child class of Manifold.
        space_args_list : list
            Arguments to pass to constructor of the manifold.
        n_points_list : list
            List of number of points on manifold to generate.
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
                ).random_point(n_points),
                rtol=rtol,
                atol=atol,
            )
            for space_args, n_points in zip(space_args_list, n_points_list)
        ]
        return self.generate_tests([], random_data)

    def _extrinsic_after_intrinsic_test_data(
        self, space_cls, space_args_list, n_points_list, rtol=gs.rtol, atol=gs.atol
    ):
        """Generate data to check that changing coordinate system twice gives back the point.

        Assumes that the first elements in space_args is the dimension of the space.

        Parameters
        ----------
        space_cls : Manifold
            Class of the space, i.e. a child class of Manifold.
        space_args_list : list
            Arguments to pass to constructor of the manifold.
        n_points_list : list
            List of number of points on manifold to generate.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for space_args, n_points in zip(space_args_list, n_points_list):

            space = space_cls(*space_args, default_coords_type="intrinsic")
            point_intrinic = space.random_point(n_points)
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
    def _compose_point_with_inverse_point_is_identity_test_data(
        self, group_cls, group_args_list, n_points_list, rtol=gs.rtol, atol=gs.atol
    ):
        """Generate data to check composition of point, inverse is identity.

        Parameters
        ----------
        group_cls : LieGroup
            Class of the group, i.e. a child class of Lie group.
        group_args_list : list
            Arguments to pass to constructor of the Lie group.
        n_points_list : list
            List of number of random points to generate.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for group_args, n_points in zip(group_args_list, n_points_list):

            group = group_cls(*group_args)
            point = group.random_point(n_points)
            random_data.append(
                dict(group_args=group_args, point=point, rtol=rtol, atol=atol)
            )

        return self.generate_tests([], random_data)

    def _compose_inverse_point_with_point_is_identity_test_data(
        self, group_cls, group_args_list, n_points_list, rtol=gs.rtol, atol=gs.atol
    ):
        """Generate data to check composition of inverse, point is identity.

        Parameters
        ----------
        group_cls : LieGroup
            Class of the group, i.e. a child class of LieGroup.
        group_args_list : list
            Arguments to pass to constructor of the Lie group.
        n_points_list : list
            List of number of random points to generate.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        return self._compose_point_with_inverse_point_is_identity_test_data(
            group_cls, group_args_list, n_points_list, rtol, atol
        )

    def _compose_point_with_identity_is_point_test_data(
        self, group_cls, group_args_list, n_points_list, rtol=gs.rtol, atol=gs.atol
    ):
        """Generate data to check composition of point, identity is point.

        Parameters
        ----------
        group_cls : LieGroup
            Class of the group, i.e. a child class of LieGroup.
        group_args_list : list
            Arguments to pass to constructor of the Lie group.
        n_points_list : list
            List of number of random points to generate.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        return self._compose_point_with_inverse_point_is_identity_test_data(
            group_cls, group_args_list, n_points_list, rtol, atol
        )

    def _compose_identity_with_point_is_point_test_data(
        self, group_cls, group_args_list, n_points_list, rtol=gs.rtol, atol=gs.atol
    ):
        """Generate data to check composition of identity, point is point.

        Parameters
        ----------
        group_cls : LieGroup
            Class of the group, i.e. a child class of LieGroup.
        group_args_list : list
            Arguments to pass to constructor of the Lie group.
        n_points_list : list
            List of number of random points to generate.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        return self._compose_point_with_inverse_point_is_identity_test_data(
            group_cls, group_args_list, n_points_list, rtol, atol
        )

    def _log_after_exp_test_data(
        self,
        group_cls,
        group_args_list,
        shape_list,
        n_tangent_vecs_list,
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
        n_tangent_vecs_list : list
            List of number of random tangent vectors to generate.
        """
        random_data = []
        for group_args, shape, n_tangent_vecs in zip(
            group_args_list, shape_list, n_tangent_vecs_list
        ):
            group = group_cls(*group_args)
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
                        rtol=rtol,
                        atol=atol,
                    )
                )

        if smoke_data is None:
            smoke_data = []
        return self.generate_tests(smoke_data, random_data)

    def _exp_after_log_test_data(
        self,
        group_cls,
        group_args_list,
        n_points_list,
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
        n_points_list : list
            List of number of points on manifold to generate.
        """
        random_data = []
        for group_args, n_points in zip(group_args_list, n_points_list):
            group = group_cls(*group_args)
            for base_point in [group.random_point(), group.identity]:
                point = group.random_point(n_points)
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

    def _to_tangent_at_identity_belongs_to_lie_algebra_test_data(
        self, group_args_list, shape_list, n_vecs_list, belongs_atol=gs.atol
    ):
        """Generate data to check that to tangent at identity belongs to lie algebra.

        Parameters
        ----------
        group_args_list : list
            List of arguments to pass to constructor of the Lie group.
        n_vecs_list : list
            List of number of vectors to be projected on tangent space at identity.
        belongs_atol : float
            Absolute tolerance of the belongs function.
        """
        random_data = []

        for group_args, shape, n_vecs in zip(group_args_list, shape_list, n_vecs_list):
            vec = gs.random.normal(size=(n_vecs,) + shape)
            random_data.append(
                dict(group_args=group_args, vec=vec, belongs_atol=belongs_atol)
            )
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
            space = self.space(*space_args)
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
    def _matrix_representation_after_basis_representation_test_data(
        self, space_cls, space_args_list, n_points_list, rtol=gs.rtol, atol=gs.atol
    ):
        """Generate data to check that changing coordinates twice gives back the point.

        Parameters
        ----------
        space_cls : LieAlgebra
            Class of the space, i.e. a child class of LieAlgebra.
        space_args_list : list
            Arguments to pass to constructor of the manifold.
        n_points_list : list
            List of number of points on manifold to generate.
        """
        random_data = [
            dict(
                space_args=space_args,
                matrix_rep=space_cls(*space_args).random_point(n_points),
                rtol=rtol,
                atol=atol,
            )
            for space_args, n_points in zip(space_args_list, n_points_list)
        ]
        return self.generate_tests([], random_data)

    def _basis_representation_after_matrix_representation_test_data(
        self, space_cls, space_args_list, n_points_list, rtol=gs.rtol, atol=gs.atol
    ):
        """Generate data to check that changing coordinates twice gives back the point.

        Parameters
        ----------
        space_cls : LieAlgebra
            Class of the space, i.e. a child class of LieAlgebra.
        space_args_list : list
            Arguments to pass to constructor of the LieAlgebra.
        n_points_list : list
            List of number of points on manifold to generate.
        """
        random_data = [
            dict(
                space_args=space_args,
                basis_rep=space_cls(*space_args).basis_representation(
                    space_cls(*space_args).random_point(n_points)
                ),
                rtol=rtol,
                atol=atol,
            )
            for space_args, n_points in zip(space_args_list, n_points_list)
        ]
        return self.generate_tests([], random_data)


class _FiberBundleTestData(TestData):
    def _is_horizontal_after_horizontal_projection_test_data(
        self, space_cls, space_args_list, n_points_list, rtol=gs.rtol, atol=gs.atol
    ):
        random_data = []
        for space_args, n_points in zip(space_args_list, n_points_list):
            space = space_cls(*space_args)
            base_point = space.random_point(n_points)
            tangent_vec = space.random_tangent_vec(base_point, n_points)
            data = dict(
                space_args=space_args,
                base_point=base_point,
                tangent_vec=tangent_vec,
                rtol=rtol,
                atol=atol,
            )
            random_data.append(data)
        return self.generate_tests([], random_data)

    def _is_vertical_after_vertical_projection_test_data(
        self, space_cls, space_args_list, n_points_list, rtol=gs.rtol, atol=gs.atol
    ):
        random_data = []
        for space_args, n_points in zip(space_args_list, n_points_list):
            space = space_cls(*space_args)
            base_point = space.random_point(n_points)
            tangent_vec = space.random_tangent_vec(base_point, n_points)
            data = dict(
                space_args=space_args,
                base_point=base_point,
                tangent_vec=tangent_vec,
                rtol=rtol,
                atol=atol,
            )
            random_data.append(data)

        return self.generate_tests([], random_data)

    def _is_horizontal_after_log_after_align_test_data(
        self,
        space_cls,
        space_args_list,
        n_points_list,
        n_base_points_list,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        random_data = [
            dict(
                space_args=space_args,
                base_point=space_cls(*space_args).random_point(n_base_points),
                point=space_cls(*space_args).random_point(n_points),
                rtol=rtol,
                atol=atol,
            )
            for space_args, n_points, n_base_points in zip(
                space_args_list, n_points_list, n_base_points_list
            )
        ]
        return self.generate_tests([], random_data)

    def _riemannian_submersion_after_lift_test_data(
        self,
        base_cls,
        space_args_list,
        n_base_points_list,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        random_data = [
            dict(
                space_args=space_args,
                base_point=base_cls(*space_args).random_point(n_points),
                rtol=rtol,
                atol=atol,
            )
            for space_args, n_points in zip(space_args_list, n_base_points_list)
        ]
        return self.generate_tests([], random_data)

    def _is_tangent_after_tangent_riemannian_submersion_test_data(
        self,
        bundle_cls,
        base_cls,
        space_args_list,
        n_vecs_list,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        random_data = []
        for space_args, n_vecs in zip(space_args_list, n_vecs_list):
            base_point = bundle_cls(*space_args).random_point()
            tangent_vec = bundle_cls(*space_args).random_tangent_vec(base_point, n_vecs)
            d = dict(
                space_args=space_args,
                base_cls=base_cls,
                tangent_vec=tangent_vec,
                base_point=base_point,
                rtol=rtol,
                atol=atol,
            )
            random_data.append(d)
        return self.generate_tests([], random_data)


class _ConnectionTestData(TestData):
    def _exp_shape_test_data(self, connection_args_list, space_list, shape_list):
        """Generate data to check that exp returns an array of the expected shape.

        Parameters
        ----------
        connection_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        shape_list : list
            List of shapes for random data to generate.
        """
        n_samples_list = [3] * len(connection_args_list)
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
                        expected=expected_shape,
                    )
                )
        return self.generate_tests([], random_data)

    def _log_shape_test_data(self, connection_args_list, space_list):
        """Generate data to check that log returns an array of the expected shape.

        Parameters
        ----------
        connection_args_list : list
            List of argument to pass to constructor of the connection.
        space_list : list
            List of manifolds on which the connection is defined.
        """
        n_samples_list = [3] * len(connection_args_list)
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
                        expected=expected_shape,
                    )
                )
        return self.generate_tests([], random_data)

    def _exp_belongs_test_data(
        self,
        connection_args_list,
        space_list,
        shape_list,
        n_tangent_vecs_list,
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
        n_tangent_vecs_list : list
            List of number of random tangent vectors to generate.
        """
        random_data = []
        for connection_args, space, shape, n_tangent_vecs in zip(
            connection_args_list, space_list, shape_list, n_tangent_vecs_list
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
        self, connection_args_list, space_list, n_points_list, is_tangent_atol=gs.atol
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

    def _exp_after_log_test_data(
        self,
        connection_args_list,
        space_list,
        n_points_list,
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
        n_points_list : list
            List of number of points on manifold to generate.
        """
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
        if smoke_data is None:
            smoke_data = []
        return self.generate_tests(smoke_data, random_data)

    def _log_after_exp_test_data(
        self,
        connection_args_list,
        space_list,
        shape_list,
        n_tangent_vecs_list,
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
        n_tangent_vecs_list : list
            List of number of random tangent vectors to generate.
        n_samples_list : list
            List of number of random data to generate.
        amplitude : float
            Factor to scale the amplitude of a tangent vector to stay in the
            injectivity domain of the exponential map.
        """
        random_data = []
        for connection_args, space, shape, n_tangent_vecs in zip(
            connection_args_list, space_list, shape_list, n_tangent_vecs_list
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
        n_tangent_vecs_list,
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
        n_tangent_vecs_list : list
            List of number of random tangent vectors to generate.
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
        for (
            connection_args,
            space,
            shape,
            n_tangent_vecs,
            n_rungs,
            alpha,
            scheme,
        ) in zip(
            connection_args_list,
            space_list,
            shape_list,
            n_tangent_vecs_list,
            n_rungs_list,
            alpha_list,
            scheme_list,
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
        n_tangent_vecs_list,
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
        n_tangent_vecs_list : list
            List of number of random tangent vectors to generate.
        n_points_list : list
            List of number of times on the geodesics.
        belongs_atol : float
            Absolute tolerance for the belongs function.
        """
        random_data = []
        for connection_args, space, shape, n_tangent_vecs, n_points in zip(
            connection_args_list,
            space_list,
            shape_list,
            n_tangent_vecs_list,
            n_points_list,
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
                    rtol=rtol,
                    atol=atol,
                )
            )
        return self.generate_tests([], random_data)


class _RiemannianMetricTestData(_ConnectionTestData):
    def _dist_is_symmetric_test_data(
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
        return self._squared_dist_is_symmetric_test_data(
            metric_args_list, space_list, n_points_a_list, n_points_b_list, rtol, atol
        )

    def _dist_is_positive_test_data(
        self,
        metric_args_list,
        space_list,
        n_points_a_list,
        n_points_b_list,
        is_positive_atol=gs.atol,
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
        is_positive_atol: float
            Absolute tolerance for checking whether value is positive.
        """
        return self._squared_dist_is_positive_test_data(
            metric_args_list,
            space_list,
            n_points_a_list,
            n_points_b_list,
            is_positive_atol,
        )

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

    def _squared_dist_is_positive_test_data(
        self,
        metric_args_list,
        space_list,
        n_points_a_list,
        n_points_b_list,
        is_positive_atol=gs.atol,
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
        is_positive_atol: float
            Absolute tolerance for checking whether value is positive.
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
                    is_positive_atol=is_positive_atol,
                )
            )
        return self.generate_tests([], random_data)

    def _dist_is_norm_of_log_test_data(
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
        return self._squared_dist_is_symmetric_test_data(
            metric_args_list, space_list, n_points_a_list, n_points_b_list, rtol, atol
        )

    def _inner_product_is_symmetric_test_data(
        self,
        metric_args_list,
        space_list,
        shape_list,
        n_tangent_vecs_list,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        """Generate data to check that the inner product is symmetric.

        Parameters
        ----------
        metric_args_list : list
            List of arguments to pass to constructor of the metric.
        space_list : list
            List of spaces on which the metric is defined.
        shape_list : list
            List of shapes for random data to generate.
        n_tangent_vecs_list : list
            List of number of random tangent vectors to generate.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for metric_args, space, shape, n_tangent_vecs in zip(
            metric_args_list, space_list, shape_list, n_tangent_vecs_list
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
                    rtol=rtol,
                    atol=atol,
                )
            )
        return self.generate_tests([], random_data)

    def _dist_point_to_itself_is_zero_test_data(
        self, metric_args_list, space_list, n_points_list, rtol=gs.rtol, atol=gs.atol
    ):
        """Generate data to check that the squared geodesic distance is symmetric.

        Parameters
        ----------
        metric_args_list : list
            List of arguments to pass to constructor of the metric.
        space_list : list
            List of spaces on which the metric is defined.
        n_points_list : list
            List of number of points to generate on the manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for metric_args, space, n_points in zip(
            metric_args_list, space_list, n_points_list
        ):
            point = space.random_point(n_points)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    point=point,
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
        n_tangent_vecs_list,
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
        n_tangent_vecs_list : list
            List of number of random tangent vectors to generate.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for metric_args, space, shape, n_tangent_vecs in zip(
            metric_args_list, space_list, shape_list, n_tangent_vecs_list
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
        n_tangent_vecs_list,
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
        n_tangent_vecs_list : list
            List of number of random tangent vectors to generate.
        is_tangent_atol: float
            Asbolute tolerance for the is_tangent function.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        random_data = []
        for metric_args, space, tangent_shape, n_tangent_vecs in zip(
            metric_args_list, space_list, shape_list, n_tangent_vecs_list
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

    def _triangle_inequality_of_dist_test_data(
        self, metric_args_list, space_list, n_points_list, atol=gs.atol
    ):
        """Generate data to check the traingle inequality of geodesic distance.

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
            metric_args_list, space_list, n_points_list
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
                    atol=atol,
                )
            )
        return self.generate_tests([], random_data)


class _InvariantMetricTestData(_RiemannianMetricTestData):
    def _exp_at_identity_of_lie_algebra_belongs_test_data(
        self, metric_args_list, group_list, n_tangent_vecs_list, belongs_atol=gs.atol
    ):
        random_data = []
        for metric_args, group, n_tangent_vecs in zip(
            metric_args_list, group_list, n_tangent_vecs_list
        ):
            lie_algebra_point = group.lie_algebra.random_point(n_tangent_vecs)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    group=group,
                    lie_algebra_point=lie_algebra_point,
                    belongs_atol=belongs_atol,
                )
            )
        return self.generate_tests([], random_data)

    def _log_at_identity_belongs_to_lie_algebra_test_data(
        self, metric_args_list, group_list, n_points_list, belongs_atol=gs.atol
    ):
        random_data = []
        for metric_args, group, n_points in zip(
            metric_args_list, group_list, n_points_list
        ):
            point = group.random_point(n_points)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    group=group,
                    point=point,
                    belongs_atol=belongs_atol,
                )
            )

        return self.generate_tests([], random_data)

    def _exp_after_log_at_identity_test_data(
        self, metric_args_list, group_list, n_points_list, rtol=gs.rtol, atol=gs.atol
    ):
        random_data = []
        for metric_args, group, n_points in zip(
            metric_args_list, group_list, n_points_list
        ):
            point = group.random_point(n_points)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    group=group,
                    point=point,
                    rtol=rtol,
                    atol=atol,
                )
            )

        return self.generate_tests([], random_data)

    def _log_after_exp_at_identity_test_data(
        self,
        metric_args_list,
        group_list,
        shape_list,
        n_tangent_vecs_list,
        amplitude=1.0,
        rtol=gs.rtol,
        atol=gs.atol,
    ):
        random_data = []

        for metric_args, group, shape, n_tangent_vecs in zip(
            metric_args_list, group_list, shape_list, n_tangent_vecs_list
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
                        rtol=rtol,
                        atol=atol,
                    )
                )
            )

        return self.generate_tests([], random_data)


class _QuotientMetricTestData(_RiemannianMetricTestData):
    def _dist_is_smaller_than_bundle_dist_test_data(
        self, metric_args_list, bundle_list, n_points_list, atol=gs.atol
    ):
        random_data = []
        for metric_args, bundle, n_points in zip(
            metric_args_list, bundle_list, n_points_list
        ):
            point_a = bundle.random_point(n_points)
            point_b = bundle.random_point(n_points)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    bundle=bundle,
                    point_a=point_a,
                    point_b=point_b,
                    atol=atol,
                )
            )
        return self.generate_tests([], random_data)

    def _log_is_horizontal_test_data(
        self, metric_args_list, bundle_list, n_points_list, atol=gs.atol
    ):
        random_data = []
        for metric_args, bundle, n_points in zip(
            metric_args_list, bundle_list, n_points_list
        ):
            point = bundle.random_point(n_points)
            base_point = bundle.random_point()
            random_data.append(
                dict(
                    metric_args=metric_args,
                    bundle=bundle,
                    point=point,
                    base_point=base_point,
                    is_horizontoal_atol=atol,
                )
            )

        return self.generate_tests([], random_data)


class _PointSetTestData(TestData):
    n_samples = 2
    n_points_list = random.sample(range(1, 5), n_samples)

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
        geom = self._SetGeometry(space)
        pts = space.random_point(2)

        dist_fnc = geom.dist

        smoke_data = [
            dict(dist_fnc=dist_fnc, point_a=pts[0], point_b=pts[1]),
            dict(dist_fnc=dist_fnc, point_a=pts[0], point_b=pts),
            dict(dist_fnc=dist_fnc, point_a=pts, point_b=pts[0]),
            dict(dist_fnc=dist_fnc, point_a=pts, point_b=pts),
        ]

        return self.generate_tests(smoke_data)

    def dist_properties_test_data(self):
        space = self._PointSet(*self.space_args_list[0])
        geom = self._SetGeometry(space)
        pts = space.random_point(3)

        dist_fnc = geom.dist

        smoke_data = [
            dict(dist_fnc=dist_fnc, point_a=pts[0], point_b=pts[1], point_c=pts[2]),
        ]

        return self.generate_tests(smoke_data)

    def geodesic_output_shape_test_data(self):
        space = self._PointSet(*self.space_args_list[0])
        geom = self._SetGeometry(space)
        pts = space.random_point(2)

        smoke_data = [
            dict(geometry=geom, start_point=pts[0], end_point=pts[0], t=0.0),
            dict(geometry=geom, start_point=pts[0], end_point=pts[0], t=[0.0, 1.0]),
            dict(geometry=geom, start_point=pts[0], end_point=pts, t=0.0),
            dict(geometry=geom, start_point=pts[0], end_point=pts, t=[0.0, 1.0]),
        ]

        return self.generate_tests(smoke_data)

    def geodesic_bounds_test_data(self):
        space = self._PointSet(*self.space_args_list[0])
        geom = self._SetGeometry(space)
        pts = space.random_point(2)

        smoke_data = [dict(geometry=geom, start_point=pts[0], end_point=pts[1])]

        return self.generate_tests(smoke_data)
