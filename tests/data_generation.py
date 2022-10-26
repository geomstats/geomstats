import copy
import itertools

import pytest

import geomstats.backend as gs
from geomstats.errors import check_parameter_accepted_values

CDTYPE = gs.get_default_cdtype()


def better_squeeze(array):
    """Delete possible singleton dimension on first axis."""
    if len(array) == 1:
        return gs.squeeze(array, axis=0)
    return array


def _expand_point(point):
    return gs.expand_dims(point, 0)


def _repeat_point(point, n_reps=2):
    if not gs.is_array(point):
        return [point] * n_reps

    return gs.repeat(_expand_point(point), n_reps, axis=0)


def _expand_and_repeat_point(point, n_reps=2):
    return _expand_point(point), _repeat_point(point, n_reps=n_reps)


def generate_random_vec(shape, dtype=gs.float64):
    """Generate a normal random vector

    Parameters
    ----------
    shape : tuple
        Shape of the vector to generate.
    dtype : dtype
        Data type of the vector to generate.

    Returns
    -------
    random_vec: array-like
        Generated random vector.
    """
    random_vec = gs.random.normal(size=shape)
    random_vec = gs.cast(random_vec, dtype=dtype)
    if dtype in [gs.complex64, gs.complex128]:
        random_vec += 1j * gs.cast(gs.random.normal(size=shape), dtype=dtype)
    return random_vec


class TestData:
    """Class for TestData objects."""

    def generate_tests(self, smoke_test_data, random_test_data=()):
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

    def _filter_combs(self, combs, vec_type, threshold):
        MAP_VEC_TYPE = {
            "repeat-first": 1,
            "repeat-second": 0,
        }
        index = MAP_VEC_TYPE[vec_type]
        other_index = (index + 1) % 2

        for comb in combs.copy():
            if comb[index] >= threshold and comb[index] != comb[other_index]:
                combs.remove(comb)

        return combs

    def _generate_datum_vectorization_tests(
        self, datum, comb_indices, arg_names, expected_name, check_expand=True, n_reps=2
    ):

        if expected_name is not None:
            has_expected = True
            expected = datum.get(expected_name)
            expected_rep = _repeat_point(expected, n_reps=n_reps)
        else:
            has_expected = False

        args_combs = []
        for arg_name in arg_names:
            arg = datum.get(arg_name)
            arg_combs = [arg]
            if check_expand:
                arg_combs.extend(_expand_and_repeat_point(arg, n_reps=n_reps))
            else:
                arg_combs.append(_repeat_point(arg, n_reps=n_reps))

            args_combs.append(arg_combs)

        new_data = []
        for indices in comb_indices:
            new_datum = copy.copy(datum)

            if has_expected:
                new_datum[expected_name] = (
                    expected_rep if (1 + int(check_expand)) in indices else expected
                )

            for arg_i, (index, arg_name) in enumerate(zip(indices, arg_names)):
                new_datum[arg_name] = args_combs[arg_i][index]

            new_data.append(new_datum)

        return new_data

    def generate_vectorization_tests(
        self,
        data,
        arg_names,
        expected_name=None,
        check_expand=True,
        n_reps=2,
        vectorization_type="sym",
    ):
        """Create new data with vectorized version of inputs.

        Parameters
        ----------
        data : list of dict
            Data. Each to vectorize.
        arg_names: list
            Name of inputs to vectorize.
        expected_name: str
            Output name in case it needs to be repeated.
        check_expand: bool
            If `True`, expanded version of each input will be tested.
        n_reps: int
            Number of times the input points should be repeated.
        vectorization_type: str
            Possible values are `sym`, `repeat-first`, `repeat-second`.
            `repeat-first` and `repeat-second` only valid for two argument case.
            `repeat-first` and `repeat-second` test asymmetric cases, repeating
            only first or second input, respectively.
        """
        check_parameter_accepted_values(
            vectorization_type,
            "vectorization_type",
            ["sym", "repeat-first", "repeat-second"],
        )

        n_args = len(arg_names)
        if n_args != 2 and vectorization_type != "sym":
            raise NotImplementedError(
                f"`{vectorization_type} only implemented for 2 arguments."
            )

        n_indices = 2 + int(check_expand)
        comb_indices = list(itertools.product(*[range(n_indices)] * len(arg_names)))
        if n_args == 2 and vectorization_type != "sym":
            comb_indices = self._filter_combs(
                comb_indices, vectorization_type, threshold=1 + int(check_expand)
            )

        new_data = []
        for datum in data:
            new_data.extend(
                self._generate_datum_vectorization_tests(
                    datum,
                    comb_indices,
                    arg_names,
                    expected_name=expected_name,
                    check_expand=check_expand,
                    n_reps=n_reps,
                )
            )

        # TODO: mark as vec?
        return self.generate_tests(new_data)


class _ManifoldTestData(TestData):
    """Class for ManifoldTestData: data to test manifold properties."""

    def manifold_shape_test_data(self):
        smoke_data = [
            dict(space_args=space_args) for space_args in self.space_args_list
        ]

        return self.generate_tests(smoke_data)

    def random_point_belongs_test_data(self):
        """Generate data to check that a random point belongs to the manifold."""
        random_data = [
            dict(space_args=space_args, n_points=n_points)
            for space_args, n_points in zip(self.space_args_list, self.n_points_list)
        ]
        return self.generate_tests([], random_data)

    def projection_belongs_test_data(self):
        """Generate data to check that a projected point belongs to the manifold."""
        random_data = [
            dict(
                space_args=space_args,
                point=gs.random.normal(size=(n_points,) + shape),
            )
            for space_args, shape, n_points in zip(
                self.space_args_list, self.shape_list, self.n_points_list
            )
        ]
        return self.generate_tests([], random_data)

    def to_tangent_is_tangent_test_data(self):
        """Generate data to check that to_tangent returns a tangent vector."""
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
                )
            )
        return self.generate_tests([], random_data)

    def random_tangent_vec_is_tangent_test_data(self):
        """Generate data to check that random tangent vec returns a tangent vector."""
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
                )
            )
        return self.generate_tests([], random_data)


class _ComplexManifoldTestData(_ManifoldTestData):
    """Class for ComplexManifoldTestData: data to test complex manifold properties."""

    def projection_belongs_test_data(self):
        """Generate data to check that a projected point belongs to the manifold."""
        random_data = [
            dict(
                space_args=space_args,
                point=generate_random_vec(shape=(n_points,) + shape, dtype=CDTYPE),
            )
            for space_args, shape, n_points in zip(
                self.space_args_list, self.shape_list, self.n_points_list
            )
        ]
        return self.generate_tests([], random_data)

    def to_tangent_is_tangent_test_data(self):
        """Generate data to check that to_tangent returns a tangent vector."""
        random_data = []

        for space_args, shape, n_vecs in zip(
            self.space_args_list, self.shape_list, self.n_vecs_list
        ):
            space = self.Space(*space_args)
            vec = generate_random_vec(shape=(n_vecs,) + shape, dtype=CDTYPE)
            base_point = space.random_point()
            random_data.append(
                dict(
                    space_args=space_args,
                    vector=vec,
                    base_point=base_point,
                )
            )
        return self.generate_tests([], random_data)


class _OpenSetTestData(_ManifoldTestData):
    def to_tangent_is_tangent_in_embedding_space_test_data(self):
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
        """Generate data to check that changing coordinate system twice
        gives back the point.

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
        """Generate data to check that changing coordinate system twice
        gives back the point.

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

    def random_point_is_tangent_test_data(self):
        """Generate data to check that random point is tangent vector."""
        random_data = []
        for space_args, n_points in zip(self.space_args_list, self.n_points_list):
            random_data += [
                dict(
                    space_args=space_args,
                    n_points=n_points,
                )
            ]

        return self.generate_tests([], random_data)

    def to_tangent_is_projection_test_data(self):
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
                )
            )
        return self.generate_tests([], random_data)


class _ComplexVectorSpaceTestData(_ComplexManifoldTestData):
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

    def random_point_is_tangent_test_data(self):
        """Generate data to check that random point is tangent vector."""
        random_data = []
        for space_args, n_points in zip(self.space_args_list, self.n_points_list):
            random_data += [
                dict(
                    space_args=space_args,
                    n_points=n_points,
                )
            ]
        return self.generate_tests([], random_data)

    def to_tangent_is_projection_test_data(self):
        """Generate data to check that to_tangent return projection."""
        random_data = []
        for space_args, shape, n_vecs in zip(
            self.space_args_list, self.shape_list, self.n_vecs_list
        ):
            space = self.Space(*space_args)
            vec = generate_random_vec(shape=(n_vecs,) + shape, dtype=CDTYPE)
            base_point = space.random_point()
            random_data.append(
                dict(
                    space_args=space_args,
                    vector=vec,
                    base_point=base_point,
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
    def manifold_shape_test_data(self):
        smoke_data = []
        for connection_args, space in zip(self.metric_args_list, self.space_list):
            smoke_data.append(
                dict(
                    connection_args=connection_args,
                    expected_shape=space.random_point().shape,
                )
            )

        return self.generate_tests(smoke_data)

    def exp_shape_test_data(self):
        """Generate data to check that exp returns an array of the expected shape."""
        n_samples_list = [3] * len(self.metric_args_list)
        random_data = []
        for connection_args, space, tangent_shape, n_samples in zip(
            self.metric_args_list, self.space_list, self.shape_list, n_samples_list
        ):
            base_point = space.random_point(n_samples)
            base_point_type = base_point.dtype
            tangent_vec = generate_random_vec(
                shape=(n_samples,) + tangent_shape, dtype=base_point_type
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
            base_point_type = base_point.dtype

            random_vec = generate_random_vec(
                shape=(n_tangent_vecs,) + shape, dtype=base_point_type
            )
            tangent_vec = space.to_tangent(random_vec, base_point)
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
            initial_point_type = initial_point.dtype
            random_vec = generate_random_vec(shape=shape, dtype=initial_point_type)
            initial_tangent_vec = space.to_tangent(random_vec, initial_point)
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
            base_point_type = base_point.dtype
            random_vec = generate_random_vec(
                shape=(n_tangent_vecs,) + shape, dtype=base_point_type
            )
            random_vec /= amplitude
            tangent_vec = space.to_tangent(random_vec, base_point)
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
            base_point_type = base_point.dtype
            random_vec = generate_random_vec(
                shape=(n_tangent_vecs,) + shape, dtype=base_point_type
            )
            random_dir = generate_random_vec(shape=shape, dtype=base_point_type)
            tangent_vec = space.to_tangent(random_vec, base_point)
            direction = space.to_tangent(random_dir, base_point)
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
        """Generate data to check the returned shape of riemann_tensor.

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
        """Generate data to check the returned shape of ricci_tensor.

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
        """Generate data to check the returned shape of scalar_curvature.

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
        """Generate data to check the returned shape of sectional_curvature.

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


class _ComplexRiemannianMetricTestData(_RiemannianMetricTestData):
    def inner_product_is_hermitian_test_data(self):
        """Generate data to check that the inner product is Hermitian."""
        random_data = []
        for metric_args, space, shape, n_tangent_vecs in zip(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        ):
            base_point = space.random_point()
            base_point_type = base_point.dtype
            random_vec_a = generate_random_vec(
                (n_tangent_vecs,) + shape, base_point_type
            )
            random_vec_b = generate_random_vec(
                (n_tangent_vecs,) + shape, base_point_type
            )
            tangent_vec_a = space.to_tangent(random_vec_a)
            tangent_vec_b = space.to_tangent(random_vec_b)
            random_data.append(
                dict(
                    metric_args=metric_args,
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
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
