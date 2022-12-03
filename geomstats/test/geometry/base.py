import abc

import pytest

import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data

# TODO: vec with tangent_vecs may not be being tested sufficiently well
# i.e. tests may pass but just because it is the repetition of points


def _get_max_ndim_point(*args):
    point_max_ndim = args[0]
    for point in args[1:]:
        if point.ndim > point_max_ndim.ndim:
            point_max_ndim = point

    return point_max_ndim


def _get_n_points(space, *args):
    point_max_ndim = _get_max_ndim_point(*args)

    if space.point_ndim == point_max_ndim.ndim:
        return 1

    return gs.prod(point_max_ndim.shape[: -space.point_ndim])


def _get_batch_shape(space, *args):
    point_max_ndim = _get_max_ndim_point(*args)
    return point_max_ndim.shape[: -space.point_ndim]


class _ProjectionTestCaseMixins:
    # TODO: should projection be part of manifold? (not only in tests)

    @abc.abstractmethod
    def _get_point_to_project(self, n_points):
        raise NotImplementedError("Need to implement `_get_point_to_project`")

    def test_projection(self, point, expected, atol):
        proj_point = self.space.projection(point)
        self.assertAllClose(proj_point, expected, atol=atol)

    @pytest.mark.vec
    def test_projection_vec(self, n_reps, atol):
        # TODO: mark as unnecessary? projection_belongs is enough?
        point = self._get_point_to_project(1)
        proj_point = self.space.projection(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=proj_point, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )

        for datum in vec_data:
            self.test_projection(**datum)

    @pytest.mark.random
    def test_projection_belongs(self, n_points, atol):
        point = self._get_point_to_project(n_points)
        proj_point = self.space.projection(point)
        expected = gs.ones(n_points, dtype=bool)

        self.test_belongs(proj_point, expected, atol)


class ManifoldTestCase(TestCase):
    # TODO: remove random_tangent_vec?
    # TODO: remove regularize
    # TODO: check default_coords_type correcteness if intrinsic by comparing
    # with point shape?

    def test_belongs(self, point, expected, atol):
        res = self.space.belongs(point, atol=atol)
        self.assertAllEqual(res, expected)

    @pytest.mark.vec
    def test_belongs_vec(self, n_reps, atol):
        # TODO: mark as unnecessary? random_point_belongs is enough?
        point = self.space.random_point()
        res = self.space.belongs(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=res, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )

        for datum in vec_data:
            self.test_belongs(**datum)

    @pytest.mark.random
    def test_not_belongs(self, n_points, atol):
        # TODO: ideally find a way to check one of each in the same output

        shape = list(self.space.shape)
        shape[0] += 1

        batch_shape = [n_points] if n_points else []
        point = gs.ones(batch_shape + shape)
        expected = gs.zeros(n_points, dtype=bool)

        self.test_belongs(point, expected, atol)

    @pytest.mark.random
    def test_random_point_belongs(self, n_points, atol):
        point = self.space.random_point(n_points)
        expected = gs.ones(n_points, dtype=bool)

        self.test_belongs(point, expected, atol)

    @pytest.mark.shape
    def test_random_point_shape(self, n_points):
        point = self.space.random_point(n_points)

        expected_ndim = self.space.point_ndim + int(n_points > 1)
        self.assertEqual(gs.ndim(point), expected_ndim)

        self.assertAllEqual(gs.shape(point)[-self.space.point_ndim :], self.space.shape)

        if n_points > 1:
            self.assertEqual(gs.shape(point)[0], n_points)

    def test_is_tangent(self, vector, base_point, expected, atol):
        res = self.space.is_tangent(vector, base_point, atol=atol)
        self.assertAllEqual(res, expected)

    @abc.abstractmethod
    def _get_vec_to_tangent(self, n_points):
        raise NotImplementedError("Need to implement `_get_vec_to_tangent`")

    @pytest.mark.vec
    def test_is_tangent_vec(self, n_reps, atol):
        vec = self._get_vec_to_tangent(1)
        point = self.space.random_point()

        tangent_vec = self.space.to_tangent(vec, point)
        res = self.space.is_tangent(tangent_vec, point)

        vec_data = generate_vectorization_data(
            data=[dict(vector=tangent_vec, base_point=point, expected=res, atol=atol)],
            arg_names=["vector", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
            vectorization_type="sym",
        )
        for datum in vec_data:
            self.test_is_tangent(**datum)

    def test_to_tangent(self, vector, base_point, expected, atol):
        tangent_vec = self.space.to_tangent(vector, base_point)
        self.assertAllClose(tangent_vec, expected, atol=atol)

    @pytest.mark.vec
    def test_to_tangent_vec(self, n_reps, atol):
        vec = self._get_vec_to_tangent(1)
        point = self.space.random_point()

        res = self.space.to_tangent(vec, point)

        vec_data = generate_vectorization_data(
            data=[dict(vector=vec, base_point=point, expected=res, atol=atol)],
            arg_names=["vector", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
            vectorization_type="sym",
        )
        for datum in vec_data:
            self.test_to_tangent(**datum)

    @pytest.mark.random
    def test_to_tangent_is_tangent(self, n_points, atol):
        vec = self._get_vec_to_tangent(n_points)
        point = self.space.random_point(n_points)

        tangent_vec = self.space.to_tangent(vec, point)
        expected = gs.ones(n_points, dtype=bool)

        self.test_is_tangent(tangent_vec, point, expected, atol)


class VectorSpaceTestCase(_ProjectionTestCaseMixins, ManifoldTestCase):
    def _get_point_to_project(self, n_points):
        return self.space.random_point(n_points)

    def _get_vec_to_tangent(self, n_points):
        return self.space.random_point(n_points)

    @pytest.mark.random
    def test_random_point_is_tangent(self, n_points, atol):
        # TODO: will we ever require a base point here?
        points = self.space.random_point(n_points)

        res = self.space.is_tangent(points, atol=atol)
        self.assertAllEqual(res, gs.ones(n_points, dtype=bool))

    @pytest.mark.random
    def test_to_tangent_is_projection(self, n_points, atol):
        vec = self.space.random_point(n_points)
        result = self.space.to_tangent(vec)
        expected = self.space.projection(vec)

        self.assertAllClose(result, expected, atol=atol)

    @pytest.mark.mathprop
    def test_basis_cardinality(self):
        basis = self.space.basis
        self.assertEqual(basis.shape[0], self.space.dim)

    @pytest.mark.mathprop
    def test_basis_belongs(self, atol):
        result = self.space.belongs(self.space.basis, atol=atol)
        self.assertAllEqual(result, gs.ones_like(result))

    def test_basis(self, expected, atol):
        self.assertAllClose(self.space.basis, expected, atol=atol)


class MatrixVectorSpaceTestCaseMixins:
    def test_to_vector(self, point, expected, atol):
        vec = self.space.to_vector(point)
        self.assertAllClose(vec, expected, atol=atol)

    @pytest.mark.vec
    def test_to_vector_vec(self, n_reps, atol):
        point = self.space.random_point()
        expected = self.space.to_vector(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )

        for datum in vec_data:
            self.test_to_vector(**datum)

    @pytest.mark.random
    def test_to_vector_and_basis(self, n_points, atol):
        mat = self.space.random_point(n_points)
        vec = self.space.to_vector(mat)

        res = gs.einsum("...i,...ijk->...jk", vec, self.space.basis)
        self.assertAllClose(res, mat, atol=atol)

    def test_from_vector(self, vec, expected, atol):
        mat = self.space.from_vector(vec)
        self.assertAllClose(mat, expected, atol=atol)

    @pytest.mark.vec
    def test_from_vector_vec(self, n_reps, atol):
        vec = gs.random.rand(self.space.dim)
        expected = self.space.from_vector(vec)

        vec_data = generate_vectorization_data(
            data=[dict(vec=vec, expected=expected, atol=atol)],
            arg_names=["vec"],
            expected_name="expected",
            n_reps=n_reps,
        )

        for datum in vec_data:
            self.test_from_vector(**datum)

    @pytest.mark.random
    def test_from_vector_belongs(self, n_points, atol):
        vec = gs.reshape(gs.random.rand(n_points * self.space.dim), (n_points, -1))
        point = self.space.from_vector(vec)

        self.test_belongs(point, gs.ones(n_points, dtype=bool), atol)

    @pytest.mark.random
    def test_from_vector_after_to_vector(self, n_points, atol):
        mat = self.space.random_point(n_points)

        vec = self.space.to_vector(mat)

        mat_ = self.space.from_vector(vec)
        self.assertAllClose(mat_, mat, atol=atol)

    @pytest.mark.random
    def test_to_vector_after_from_vector(self, n_points, atol):
        vec = gs.reshape(gs.random.rand(n_points * self.space.dim), (n_points, -1))

        mat = self.space.from_vector(vec)

        vec_ = self.space.to_vector(mat)
        self.assertAllClose(vec_, vec, atol=atol)


class MatrixLieAlgebraTestCase(VectorSpaceTestCase):
    # most of tests are very similar to MatrixVectorSpaceTestCaseMixins

    # TODO: any mathematical property for baker_campbell_hausdorff?

    def test_baker_campbell_hausdorff(
        self, matrix_a, matrix_b, expected, atol, order=2
    ):
        res = self.space.baker_campbell_hausdorff(matrix_a, matrix_b, order=order)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_baker_campbell_hausdorff_vec(self, n_reps, atol, order=2):
        matrix_a, matrix_b = self.space.random_point(2)
        expected = self.space.baker_campbell_hausdorff(matrix_a, matrix_b, order=order)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    matrix_a=matrix_a,
                    matrix_b=matrix_b,
                    expected=expected,
                    atol=atol,
                    order=order,
                )
            ],
            arg_names=["matrix_a", "matrix_b"],
            expected_name="expected",
            n_reps=n_reps,
            vectorization_type="sym",
        )
        for datum in vec_data:
            self.test_baker_campbell_hausdorff(**datum)

    def test_basis_representation(self, point, expected, atol):
        vec = self.space.basis_representation(point)
        self.assertAllClose(vec, expected, atol=atol)

    @pytest.mark.vec
    def test_basis_representation_vec(self, n_reps, atol):
        point = self.space.random_point()
        expected = self.space.basis_representation(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )

        for datum in vec_data:
            self.test_basis_representation(**datum)

    @pytest.mark.random
    def test_basis_representation_and_basis(self, n_points, atol):
        mat = self.space.random_point(n_points)
        vec = self.space.basis_representation(mat)

        res = gs.einsum("...i,...ijk->...jk", vec, self.space.basis)
        self.assertAllClose(res, mat, atol=atol)

    def test_matrix_representation(self, vec, expected, atol):
        mat = self.space.matrix_representation(vec)
        self.assertAllClose(mat, expected, atol=atol)

    @pytest.mark.vec
    def test_matrix_representation_vec(self, n_reps, atol):
        vec = gs.random.rand(self.space.dim)
        expected = self.space.matrix_representation(vec)

        vec_data = generate_vectorization_data(
            data=[dict(vec=vec, expected=expected, atol=atol)],
            arg_names=["vec"],
            expected_name="expected",
            n_reps=n_reps,
        )

        for datum in vec_data:
            self.test_matrix_representation(**datum)

    @pytest.mark.random
    def test_matrix_representation_belongs(self, n_points, atol):
        vec = gs.reshape(gs.random.rand(n_points * self.space.dim), (n_points, -1))
        point = self.space.matrix_representation(vec)

        self.test_belongs(point, gs.ones(n_points, dtype=bool), atol)

    @pytest.mark.random
    def test_matrix_representation_after_basis_representation(self, n_points, atol):
        point = self.space.random_point(n_points)
        vec = self.space.basis_representation(point)
        point_ = self.space.matrix_representation(vec)

        self.assertAllClose(point_, point, atol=atol)

    @pytest.mark.random
    def test_basis_representation_after_matrix_representation(self, n_points, atol):
        vec = gs.reshape(gs.random.rand(n_points * self.space.dim), (n_points, -1))
        point = self.space.matrix_representation(vec)
        vec_ = self.space.basis_representation(point)

        self.assertAllClose(vec_, vec, atol=atol)


class LevelSetTestCase(_ProjectionTestCaseMixins, ManifoldTestCase):
    # TODO: need to develop `intrinsic_after_extrinsic` and `extrinsic_after_intrinsic`
    # TODO: class to handle `extrinsinc-intrinsic` mixins?

    def _get_point_to_project(self, n_points):
        return self.space.embedding_space.random_point(n_points)

    def _get_vec_to_tangent(self, n_points):
        return self.space.embedding_space.random_point(n_points)

    def test_submersion(self, point, expected, atol):
        submersed_point = self.space.submersion(point)
        self.assertAllClose(submersed_point, expected, atol=atol)

    def test_submersion_is_zero(self, point, submersion_shape, atol):
        # TODO: keep?
        batch_shape = _get_batch_shape(self.space, point)
        expected = gs.zeros(batch_shape + submersion_shape)

        self.test_submersion(point, expected, atol)

    @pytest.mark.vec
    def test_submersion_vec(self, n_reps, atol):
        # TODO: mark as redundant? belongs suffices?
        point = self.space.embedding_space.random_point()
        expected = self.space.submersion(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )

        for datum in vec_data:
            self.test_submersion(**datum)

    def test_tangent_submersion(self, vector, point, expected, atol):
        submersed_vector = self.space.tangent_submersion(vector, point)
        self.assertAllClose(submersed_vector, expected, atol=atol)

    def test_tangent_submersion_is_zero(
        self, tangent_vector, point, tangent_submersion_shape, atol
    ):
        # TODO: keep?
        batch_shape = _get_batch_shape(self.space, tangent_vector, point)
        expected = gs.zeros(batch_shape + tangent_submersion_shape)

        self.test_tangent_submersion(tangent_vector, point, expected, atol)

    @pytest.mark.vec
    def test_tangent_submersion_vec(self, n_reps, atol):
        # TODO: mark as redundant? is_tangent suffices?
        vector, point = self.space.random_point(2)
        expected = self.space.tangent_submersion(vector, point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    vector=vector,
                    point=point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["vector", "point"],
            n_reps=n_reps,
            vectorization_type="sym",
            expected_name="expected",
        )

        for datum in vec_data:
            self.test_tangent_submersion(**datum)
