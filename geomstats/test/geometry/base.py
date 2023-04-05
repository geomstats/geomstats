import abc

import pytest

import geomstats.backend as gs
from geomstats.test.random import (
    FiberBundleRandomDataGenerator,
    RandomDataGenerator,
    get_random_times,
)
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.vectorization import get_batch_shape, repeat_point

# TODO: vec with tangent_vecs may not be being tested sufficiently well
# i.e. tests may pass but just because it is the repetition of points

# TODO: define better where to use pytes.mark.mathprop

# TODO: enumerate tests for which random is not enough

# TODO: review uses of gs.ones and gs.zeros

# TODO: review "passes" in tests (maybe not implemented error?)


class _ProjectionTestCaseMixins:
    # TODO: should projection be part of manifold? (not only in tests)

    @abc.abstractmethod
    def _get_point_to_project(self, n_points=1):
        raise NotImplementedError("Need to implement `_get_point_to_project`")

    def test_projection(self, point, expected, atol):
        proj_point = self.space.projection(point)
        self.assertAllClose(proj_point, expected, atol=atol)

    @pytest.mark.vec
    def test_projection_vec(self, n_reps, atol):
        point = self._get_point_to_project(1)
        expected = self.space.projection(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_projection_belongs(self, n_points, atol):
        point = self._get_point_to_project(n_points)
        proj_point = self.space.projection(point)
        expected = gs.ones(n_points, dtype=bool)

        self.test_belongs(proj_point, expected, atol)


class _LieGroupTestCaseMixins:
    def test_compose(self, point_a, point_b, expected, atol):
        composed = self.space.compose(point_a, point_b)
        self.assertAllClose(composed, expected, atol=atol)

    @pytest.mark.vec
    def test_compose_vec(self, n_reps, atol):
        point_a, point_b = self.data_generator.random_point(2)

        expected = self.space.compose(point_a, point_b)

        vec_data = generate_vectorization_data(
            data=[dict(point_a=point_a, point_b=point_b, expected=expected, atol=atol)],
            arg_names=["point_a", "point_b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_inverse(self, point, expected, atol):
        inverse = self.space.inverse(point)
        self.assertAllClose(inverse, expected, atol=atol)

    @pytest.mark.vec
    def test_inverse_vec(self, n_reps, atol):
        point = self.data_generator.random_point()

        expected = self.space.inverse(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_compose_with_inverse_is_identity(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        inverse = self.space.inverse(point)

        identity = self.space.identity
        if n_points > 1:
            identity = gs.broadcast_to(identity, (n_points, *identity.shape))

        identity_ = self.space.compose(point, inverse)
        self.assertAllClose(identity_, identity, atol=atol)

        identity_ = self.space.compose(inverse, point)
        self.assertAllClose(identity_, identity, atol=atol)

    @pytest.mark.random
    def test_compose_with_identity_is_point(self, n_points, atol):
        point = self.data_generator.random_point(n_points)

        point_ = self.space.compose(point, self.space.identity)
        self.assertAllClose(point_, point, atol=atol)

        point_ = self.space.compose(self.space.identity, point)
        self.assertAllClose(point_, point, atol=atol)

    def test_exp(self, tangent_vec, base_point, expected, atol):
        point = self.space.exp(tangent_vec, base_point)
        self.assertAllClose(point, expected, atol=atol)

    @pytest.mark.vec
    def test_exp_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.exp(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_log(self, point, base_point, expected, atol):
        vec = self.space.log(point, base_point)
        self.assertAllClose(vec, expected, atol=atol)

    @pytest.mark.vec
    def test_log_vec(self, n_reps, atol):
        point, base_point = self.data_generator.random_point(2)

        expected = self.space.log(point, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(point=point, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["point", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_exp_after_log(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        base_point = self.data_generator.random_point(n_points)

        vec = self.space.log(point, base_point)
        point_ = self.space.exp(vec, base_point)

        self.assertAllClose(point_, point, atol=atol)

    @pytest.mark.random
    def test_log_after_exp(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        point = self.space.exp(tangent_vec, base_point)
        tangent_vec_ = self.space.log(point, base_point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    @pytest.mark.random
    def test_to_tangent_at_identity_belongs_to_lie_algebra(self, n_points, atol):
        tangent_vec = self.data_generator.random_tangent_vec(
            repeat_point(self.space.identity, n_points)
        )

        res = self.space.lie_algebra.belongs(tangent_vec, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    def test_tangent_translation_map(
        self, point, left, inverse, tangent_vec, expected, atol
    ):
        res = self.space.tangent_translation_map(point, left=left, inverse=inverse)(
            tangent_vec
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_tangent_translation_map_vec(self, n_reps, left, inverse, atol):
        point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(point)

        expected = self.space.tangent_translation_map(
            point, left=left, inverse=inverse
        )(tangent_vec)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    point=point,
                    tangent_vec=tangent_vec,
                    left=left,
                    inverse=inverse,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["point", "tangent_vec"],
            expected_name="expected",
            vectorization_type="repeat-1",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_lie_bracket(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        # TODO: any random test for validation here?
        bracket = self.space.lie_bracket(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(bracket, expected, atol=atol)

    @pytest.mark.vec
    def test_lie_bracket_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a, tangent_vec_b = self.data_generator.random_tangent_vec(
            repeat_point(base_point, 2)
        )

        expected = self.space.lie_bracket(tangent_vec_a, tangent_vec_b, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)


class _ManifoldTestCaseMixins:
    # TODO: remove random_tangent_vec?
    # TODO: remove regularize
    # TODO: check default_coords_type correcteness if intrinsic by comparing
    # with point shape?

    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    def test_belongs(self, point, expected, atol):
        res = self.space.belongs(point, atol=atol)
        self.assertAllEqual(res, expected)

    @pytest.mark.vec
    def test_belongs_vec(self, n_reps, atol):
        # TODO: mark as unnecessary? random_point_belongs is enough?
        point = self.data_generator.random_point()
        res = self.space.belongs(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=res, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

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
        point = self.data_generator.random_point(n_points)
        expected = gs.ones(n_points, dtype=bool)

        self.test_belongs(point, expected, atol)

    @pytest.mark.shape
    def test_random_point_shape(self, n_points):
        point = self.data_generator.random_point(n_points)

        expected_ndim = self.space.point_ndim + int(n_points > 1)
        self.assertEqual(gs.ndim(point), expected_ndim)

        self.assertAllEqual(gs.shape(point)[-self.space.point_ndim :], self.space.shape)

        if n_points > 1:
            self.assertEqual(gs.shape(point)[0], n_points)

    def test_is_tangent(self, vector, base_point, expected, atol):
        res = self.space.is_tangent(vector, base_point, atol=atol)
        self.assertAllEqual(res, expected)

    @pytest.mark.vec
    def test_is_tangent_vec(self, n_reps, atol):
        point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(point)

        res = self.space.is_tangent(tangent_vec, point)

        vec_data = generate_vectorization_data(
            data=[dict(vector=tangent_vec, base_point=point, expected=res, atol=atol)],
            arg_names=["vector", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_to_tangent(self, vector, base_point, expected, atol):
        tangent_vec = self.space.to_tangent(vector, base_point)
        self.assertAllClose(tangent_vec, expected, atol=atol)

    @pytest.mark.vec
    def test_to_tangent_vec(self, n_reps, atol):
        # TODO: check if it makes sense
        vec = self.data_generator.random_point()
        point = self.data_generator.random_point()

        res = self.space.to_tangent(vec, point)

        vec_data = generate_vectorization_data(
            data=[dict(vector=vec, base_point=point, expected=res, atol=atol)],
            arg_names=["vector", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_to_tangent_is_tangent(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(point)

        expected = gs.ones(n_points, dtype=bool)

        self.test_is_tangent(tangent_vec, point, expected, atol)

    def test_regularize(self, point, expected, atol):
        regularized_point = self.space.regularize(point)
        self.assertAllClose(regularized_point, expected, atol=atol)

    @pytest.mark.vec
    def test_regularize_vec(self, n_reps, atol):
        point = self.data_generator.random_point()
        expected = self.space.regularize(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)


class ManifoldTestCase(_ManifoldTestCaseMixins, TestCase):
    pass


class ComplexManifoldTestCase(_ManifoldTestCaseMixins, TestCase):
    # TODO: add random_point_is_complex
    # TODO: check imaginary part of random_point

    @pytest.mark.type
    def test_random_point_is_complex(self, n_points):
        point = self.data_generator.random_point(n_points)

        self.assertTrue(gs.is_complex(point))

    @pytest.mark.random
    def test_random_point_imaginary_nonzero(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        res = gs.imag(gs.abs(point))
        self.assertAllClose(res, gs.zeros_like(point), atol=atol)


class _VectorSpaceTestCaseMixins(_ProjectionTestCaseMixins):
    def _get_point_to_project(self, n_points):
        return self.data_generator.random_point(n_points)

    @pytest.mark.random
    def test_random_point_is_tangent(self, n_points, atol):
        # TODO: will we ever require a base point here?
        points = self.data_generator.random_point(n_points)

        res = self.space.is_tangent(points, atol=atol)
        self.assertAllEqual(res, gs.ones(n_points, dtype=bool))

    @pytest.mark.random
    def test_to_tangent_is_projection(self, n_points, atol):
        vec = self.data_generator.random_point(n_points)
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


class VectorSpaceTestCase(_VectorSpaceTestCaseMixins, ManifoldTestCase):
    pass


class ComplexVectorSpaceTestCase(_VectorSpaceTestCaseMixins, ComplexManifoldTestCase):
    pass


class MatrixVectorSpaceTestCaseMixins:
    def _get_random_vector(self, n_points=1):
        # TODO: from data generator?
        if n_points == 1:
            return gs.random.rand(self.space.dim)

        return gs.reshape(gs.random.rand(n_points * self.space.dim), (n_points, -1))

    def test_to_vector(self, point, expected, atol):
        vec = self.space.to_vector(point)
        self.assertAllClose(vec, expected, atol=atol)

    @pytest.mark.vec
    def test_to_vector_vec(self, n_reps, atol):
        point = self.data_generator.random_point()
        expected = self.space.to_vector(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_to_vector_and_basis(self, n_points, atol):
        mat = self.data_generator.random_point(n_points)
        vec = self.space.to_vector(mat)

        res = gs.einsum("...i,...ijk->...jk", vec, self.space.basis)
        self.assertAllClose(res, mat, atol=atol)

    def test_from_vector(self, vec, expected, atol):
        mat = self.space.from_vector(vec)
        self.assertAllClose(mat, expected, atol=atol)

    @pytest.mark.vec
    def test_from_vector_vec(self, n_reps, atol):
        vec = self._get_random_vector()
        expected = self.space.from_vector(vec)

        vec_data = generate_vectorization_data(
            data=[dict(vec=vec, expected=expected, atol=atol)],
            arg_names=["vec"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_from_vector_belongs(self, n_points, atol):
        vec = self._get_random_vector(n_points)
        point = self.space.from_vector(vec)

        self.test_belongs(point, gs.ones(n_points, dtype=bool), atol)

    @pytest.mark.random
    def test_from_vector_after_to_vector(self, n_points, atol):
        mat = self.data_generator.random_point(n_points)

        vec = self.space.to_vector(mat)

        mat_ = self.space.from_vector(vec)
        self.assertAllClose(mat_, mat, atol=atol)

    @pytest.mark.random
    def test_to_vector_after_from_vector(self, n_points, atol):
        vec = self._get_random_vector(n_points)

        mat = self.space.from_vector(vec)

        vec_ = self.space.to_vector(mat)
        self.assertAllClose(vec_, vec, atol=atol)


class ComplexMatrixVectorSpaceTestCaseMixins(MatrixVectorSpaceTestCaseMixins):
    def _get_random_vector(self, n_points=1):
        if n_points == 1:
            return gs.random.rand(self.space.dim, dtype=gs.get_default_cdtype())

        return gs.reshape(
            gs.random.rand(n_points * self.space.dim),
            (n_points, -1),
            dtype=gs.get_default_cdtype(),
        )


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
        matrix_a, matrix_b = self.data_generator.random_point(2)
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
        )
        self._test_vectorization(vec_data)

    def test_basis_representation(self, point, expected, atol):
        vec = self.space.basis_representation(point)
        self.assertAllClose(vec, expected, atol=atol)

    @pytest.mark.vec
    def test_basis_representation_vec(self, n_reps, atol):
        point = self.data_generator.random_point()
        expected = self.space.basis_representation(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_basis_representation_and_basis(self, n_points, atol):
        mat = self.data_generator.random_point(n_points)
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
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_matrix_representation_belongs(self, n_points, atol):
        vec = gs.reshape(gs.random.rand(n_points * self.space.dim), (n_points, -1))
        point = self.space.matrix_representation(vec)

        self.test_belongs(point, gs.ones(n_points, dtype=bool), atol)

    @pytest.mark.random
    def test_matrix_representation_after_basis_representation(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        vec = self.space.basis_representation(point)
        point_ = self.space.matrix_representation(vec)

        self.assertAllClose(point_, point, atol=atol)

    @pytest.mark.random
    def test_basis_representation_after_matrix_representation(self, n_points, atol):
        vec = gs.reshape(gs.random.rand(n_points * self.space.dim), (n_points, -1))
        point = self.space.matrix_representation(vec)
        vec_ = self.space.basis_representation(point)

        self.assertAllClose(vec_, vec, atol=atol)


class MatrixLieGroupTestCase(_LieGroupTestCaseMixins, ManifoldTestCase):
    pass


class LieGroupTestCase(_LieGroupTestCaseMixins, ManifoldTestCase):
    # TODO: exp and log not from identity: are they enough tested with log and exp?

    def test_jacobian_translation(self, point, left, expected, atol):
        res = self.space.jacobian_translation(point, left=left)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_jacobian_translation_vec(self, n_reps, left, atol):
        point = self.data_generator.random_point()
        expected = self.space.jacobian_translation(point, left=left)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, left=left, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_exp_from_identity(self, tangent_vec, expected, atol):
        res = self.space.exp_from_identity(tangent_vec)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_exp_from_identity_vec(self, n_reps, atol):
        tangent_vec = self.data_generator.random_tangent_vec(self.space.identity)
        expected = self.space.exp_from_identity(tangent_vec)

        vec_data = generate_vectorization_data(
            data=[dict(tangent_vec=tangent_vec, expected=expected, atol=atol)],
            arg_names=["tangent_vec"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_log_from_identity(self, point, expected, atol):
        vec = self.space.log_from_identity(point)
        self.assertAllClose(vec, expected, atol=atol)

    @pytest.mark.vec
    def test_log_from_identity_vec(self, n_reps, atol):
        point = self.data_generator.random_point()
        expected = self.space.log_from_identity(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_exp_from_identity_after_log_from_identity(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        tangent_vec = self.space.log_from_identity(point)

        point_ = self.space.exp_from_identity(tangent_vec)
        self.assertAllClose(point_, point, atol=atol)

    @pytest.mark.random
    def test_log_from_identity_after_exp_from_identity(self, n_points, atol):
        tangent_vec = self.data_generator.random_tangent_vec(
            repeat_point(self.space.identity, n_points)
        )

        point = self.space.exp_from_identity(tangent_vec)
        tangent_vec_ = self.space.log_from_identity(point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)


class LevelSetTestCase(_ProjectionTestCaseMixins, ManifoldTestCase):
    # TODO: need to develop `intrinsic_after_extrinsic` and `extrinsic_after_intrinsic`
    # TODO: class to handle `extrinsinc-intrinsic` mixins?

    def _get_point_to_project(self, n_points):
        return self.space.embedding_space.random_point(n_points)

    def test_submersion(self, point, expected, atol):
        submersed_point = self.space.submersion(point)
        self.assertAllClose(submersed_point, expected, atol=atol)

    def test_submersion_is_zero(self, point, submersion_shape, atol):
        # TODO: keep?
        batch_shape = get_batch_shape(self.space, point)
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
        self._test_vectorization(vec_data)

    def test_tangent_submersion(self, vector, point, expected, atol):
        submersed_vector = self.space.tangent_submersion(vector, point)
        self.assertAllClose(submersed_vector, expected, atol=atol)

    def test_tangent_submersion_is_zero(
        self, tangent_vector, point, tangent_submersion_shape, atol
    ):
        # TODO: keep?
        batch_shape = get_batch_shape(self.space, tangent_vector, point)
        expected = gs.zeros(batch_shape + tangent_submersion_shape)

        self.test_tangent_submersion(tangent_vector, point, expected, atol)

    @pytest.mark.vec
    def test_tangent_submersion_vec(self, n_reps, atol):
        # TODO: mark as redundant? is_tangent suffices?
        vector, point = self.data_generator.random_point(2)
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
            expected_name="expected",
        )
        self._test_vectorization(vec_data)


class _OpenSetTestCaseMixins(_ProjectionTestCaseMixins):
    def _get_point_to_project(self, n_points):
        return self.space.embedding_space.random_point(n_points)

    @pytest.mark.random
    def test_to_tangent_is_tangent_in_embedding_space(self, n_points):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        res = self.space.embedding_space.is_tangent(tangent_vec, base_point)

        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)


class OpenSetTestCase(_OpenSetTestCaseMixins, ManifoldTestCase):
    pass


class ComplexOpenSetTestCase(_OpenSetTestCaseMixins, ComplexManifoldTestCase):
    pass


class FiberBundleTestCase(ManifoldTestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = FiberBundleRandomDataGenerator(self.space, self.base)

    def _test_belongs_to_base(self, point, expected, atol):
        res = self.base.belongs(point, atol=atol)
        self.assertAllEqual(res, expected)

    def test_riemannian_submersion(self, point, expected, atol):
        res = self.space.riemannian_submersion(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_riemannian_submersion_vec(self, n_reps, atol):
        point = self.data_generator.random_point()
        expected = self.space.riemannian_submersion(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_riemannian_submersion_belongs_to_base(self, n_points, atol):
        point = self.data_generator.random_point(n_points)

        proj_point = self.space.riemannian_submersion(point)
        expected = gs.ones(n_points, dtype=bool)

        self._test_belongs_to_base(proj_point, expected, atol)

    def test_lift(self, point, expected, atol):
        res = self.space.lift(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_lift_vec(self, n_reps, atol):
        point = self.data_generator.base_random_point()
        expected = self.space.lift(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_lift_belongs_to_total_space(self, n_points, atol):
        point = self.data_generator.base_random_point(n_points)
        lifted_point = self.space.lift(point)

        expected = gs.ones(n_points, dtype=bool)
        self.test_belongs(lifted_point, expected, atol)

    @pytest.mark.random
    def test_riemannian_submersion_after_lift(self, n_points, atol):
        point = self.data_generator.base_random_point(n_points)
        lifted_point = self.space.lift(point)
        point_ = self.space.riemannian_submersion(lifted_point)

        self.assertAllClose(point_, point, atol=atol)

    def test_tangent_riemannian_submersion(
        self, tangent_vec, base_point, expected, atol
    ):
        res = self.space.tangent_riemannian_submersion(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_tangent_riemannian_submersion_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.tangent_riemannian_submersion(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_tangent_riemannian_submersion_is_tangent(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        proj_tangent_vector = self.space.tangent_riemannian_submersion(
            tangent_vec, base_point
        )
        proj_point = self.space.riemannian_submersion(base_point)

        res = self.base.is_tangent(proj_tangent_vector, proj_point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    def test_align(self, point, base_point, expected, atol):
        res = self.space.align(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_align_vec(self, n_reps, atol):
        point = self.data_generator.random_point()
        base_point = self.data_generator.random_point()

        expected = self.space.align(point, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(point=point, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["point", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_log_after_align_is_horizontal(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        base_point = self.data_generator.random_point(n_points)

        aligned_point = self.space.align(point, base_point)
        log = self.space.metric.log(aligned_point, base_point)

        expected = gs.ones(n_points, dtype=bool)
        self.test_is_horizontal(log, base_point, expected, atol)

    def test_horizontal_projection(self, tangent_vec, base_point, expected, atol):
        res = self.space.horizontal_projection(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_horizontal_projection_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.horizontal_projection(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_horizontal_projection_is_horizontal(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        horizontal = self.space.horizontal_projection(tangent_vec, base_point)
        expected = gs.ones(n_points, dtype=bool)
        self.test_is_horizontal(horizontal, base_point, expected, atol)

    def test_vertical_projection(self, tangent_vec, base_point, expected, atol):
        res = self.space.vertical_projection(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_vertical_projection_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.vertical_projection(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
            vectorization_type="repeat-0",
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_vertical_projection_is_vertical(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        vertical = self.space.vertical_projection(tangent_vec, base_point)
        expected = gs.ones(n_points, dtype=bool)
        self.test_is_vertical(vertical, base_point, expected, atol)

    @pytest.mark.random
    def test_tangent_riemannian_submersion_after_vertical_projection(
        self, n_points, atol
    ):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        vertical = self.space.vertical_projection(tangent_vec, base_point)
        res = self.space.tangent_riemannian_submersion(vertical, base_point)
        expected = gs.zeros_like(res)

        self.assertAllClose(res, expected, atol=atol)

    def test_is_horizontal(self, tangent_vec, base_point, expected, atol):
        res = self.space.is_horizontal(tangent_vec, base_point, atol=atol)
        self.assertAllEqual(res, expected)

    @pytest.mark.vec
    def test_is_horizontal_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.is_horizontal(tangent_vec, base_point, atol=atol)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_is_vertical(self, tangent_vec, base_point, expected, atol):
        res = self.space.is_vertical(tangent_vec, base_point, atol=atol)
        self.assertAllEqual(res, expected)

    @pytest.mark.vec
    def test_is_vertical_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.is_vertical(tangent_vec, base_point, atol=atol)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_horizontal_lift(
        self, tangent_vec, expected, atol, base_point=None, fiber_point=None
    ):
        res = self.space.horizontal_lift(
            tangent_vec, base_point=base_point, fiber_point=fiber_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_horizontal_lift_vec(self, n_reps, atol):
        fiber_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(fiber_point)

        expected = self.space.horizontal_lift(tangent_vec, fiber_point=fiber_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    fiber_point=fiber_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "fiber_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_horizontal_lift_is_horizontal(self, n_points, atol):
        fiber_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(fiber_point)

        horizontal = self.space.horizontal_lift(tangent_vec, fiber_point=fiber_point)
        expected = gs.ones(n_points, dtype=bool)
        self.test_is_horizontal(horizontal, fiber_point, expected, atol)

    @pytest.mark.random
    def test_tangent_riemannian_submersion_after_horizontal_lift(self, n_points, atol):
        base_point = self.data_generator.base_random_point(n_points)
        tangent_vec = self.data_generator.base_random_tangent_vec(base_point)
        fiber_point = self.space.lift(base_point)

        horizontal = self.space.horizontal_lift(tangent_vec, fiber_point=fiber_point)
        tangent_vec_ = self.space.tangent_riemannian_submersion(horizontal, fiber_point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    def test_integrability_tensor(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        res = self.space.integrability_tensor(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_integrability_tensor_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.integrability_tensor(
            tangent_vec_a, tangent_vec_b, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_integrability_tensor_derivative(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        nabla_x_y,
        tangent_vec_e,
        nabla_x_e,
        base_point,
        expected_nabla_x_a_y_e,
        expected_a_y_e,
        atol,
    ):
        nabla_x_a_y_e, a_y_e = self.space.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            nabla_x_y,
            tangent_vec_e,
            nabla_x_e,
            base_point,
        )
        self.assertAllClose(nabla_x_a_y_e, expected_nabla_x_a_y_e, atol=atol)
        self.assertAllClose(a_y_e, expected_a_y_e, atol=atol)

    @pytest.mark.vec
    def test_integrability_tensor_derivative_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        horizontal_vec_x = self.space.horizontal_lift(
            self.data_generator.random_tangent_vec(base_point),
            fiber_point=base_point,
        )
        horizontal_vec_y = self.space.horizontal_lift(
            self.data_generator.random_tangent_vec(base_point),
            fiber_point=base_point,
        )
        nabla_x_y = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_e = self.data_generator.random_tangent_vec(base_point)
        nabla_x_e = self.data_generator.random_tangent_vec(base_point)

        nabla_x_a_y_e, a_y_e = self.space.integrability_tensor_derivative(
            horizontal_vec_x,
            horizontal_vec_y,
            nabla_x_y,
            tangent_vec_e,
            nabla_x_e,
            base_point,
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    horizontal_vec_x=horizontal_vec_x,
                    horizontal_vec_y=horizontal_vec_y,
                    nabla_x_y=nabla_x_y,
                    tangent_vec_e=tangent_vec_e,
                    nabla_x_e=nabla_x_e,
                    base_point=base_point,
                    expected_nabla_x_a_y_e=nabla_x_a_y_e,
                    expected_a_y_e=a_y_e,
                    atol=atol,
                )
            ],
            arg_names=[
                "horizontal_vec_x",
                "horizontal_vec_y",
                "nabla_x_y",
                "tangent_vec_e",
                "nabla_x_e",
                "base_point",
            ],
            expected_name=["expected_nabla_x_a_y_e", "expected_a_y_e"],
            n_reps=n_reps,
            vectorization_type="basic",
        )
        self._test_vectorization(vec_data)


class ConnectionTestCase(TestCase):
    # TODO: exp_after_log
    # TODO: log_after_exp
    # TODO: geodesic and inverse parametrization geodesic
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    def test_christoffels(self, base_point, expected, atol):
        res = self.space.metric.christoffels(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_christoffels_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.christoffels(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_exp(self, tangent_vec, base_point, expected, atol):
        res = self.space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_exp_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.exp(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_log(self, point, base_point, expected, atol):
        res = self.space.metric.log(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_log_vec(self, n_reps, atol):
        point, base_point = self.data_generator.random_point(2)

        expected = self.space.metric.log(point, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(point=point, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["point", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_riemann_tensor(self, base_point, expected, atol):
        res = self.space.metric.riemann_tensor(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_riemann_tensor_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.riemann_tensor(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_curvature(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point, expected, atol
    ):
        res = self.space.metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_curvature_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_c = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    tangent_vec_c=tangent_vec_c,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b", "tangent_vec_c", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_ricci_tensor(self, base_point, expected, atol):
        res = self.space.metric.ricci_tensor(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_ricci_tensor_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.ricci_tensor(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_directional_curvature(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        res = self.space.metric.directional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_directional_curvature_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.directional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_curvature_derivative(
        self,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        tangent_vec_d,
        base_point,
        expected,
        atol,
    ):
        res = self.space.metric.curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_curvature_derivative_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_c = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_d = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    tangent_vec_c=tangent_vec_c,
                    tangent_vec_d=tangent_vec_d,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=[
                "tangent_vec_a",
                "tangent_vec_b",
                "tangent_vec_c",
                "tangent_vec_d",
                "base_point",
            ],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_directional_curvature_derivative(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        res = self.space.metric.directional_curvature_derivative(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_directional_curvature_derivative_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.directional_curvature_derivative(
            tangent_vec_a, tangent_vec_b, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_geodesic(self, initial_point, end_point, time, expected, atol):
        res = self.space.metric.geodesic(initial_point, end_point=end_point)(time)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_geodesic_vec(self, n_reps, n_times, atol):
        initial_point, end_point = self.data_generator.random_point(2)
        time = get_random_times(n_times)

        expected = self.space.metric.geodesic(initial_point, end_point=end_point)(time)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    initial_point=initial_point,
                    end_point=end_point,
                    time=time,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["initial_point", "end_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_parallel_transport(
        self, tangent_vec, base_point, expected, atol, direction=None, end_point=None
    ):
        res = self.space.metric.parallel_transport(
            tangent_vec,
            base_point,
            direction=direction,
            end_point=end_point,
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_parallel_transport_vec_with_direction(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        direction = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.parallel_transport(
            tangent_vec, base_point, direction=direction
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    direction=direction,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point", "direction"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data, test_fnc_name="test_parallel_transport")

    @pytest.mark.vec
    def test_parallel_transport_vec_with_end_point(self, n_reps, atol):
        base_point, end_point = self.data_generator.random_point(2)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    end_point=end_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point", "end_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data, test_fnc_name="test_parallel_transport")

    def test_injectivity_radius(self, base_point, expected, atol):
        res = self.space.metric.injectivity_radius(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_injectivity_radius_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.injectivity_radius(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)


class RiemannianMetricTestCase(ConnectionTestCase):

    # TODO: dist is positive
    # TODO: dist is symmetric

    def test_metric_matrix(self, base_point, expected, atol):
        res = self.space.metric.metric_matrix(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_metric_matrix_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.metric_matrix(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_cometric_matrix(self, base_point, expected, atol):
        res = self.space.metric.cometric_matrix(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_cometric_matrix_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.cometric_matrix(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_inner_product_derivative_matrix(self, base_point, expected, atol):
        res = self.space.metric.inner_product_derivative_matrix(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inner_product_derivative_matrix_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.inner_product_derivative_matrix(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_inner_product(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        res = self.space.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inner_product_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_inner_coproduct(
        self, cotangent_vec_a, cotangent_vec_b, base_point, expected, atol
    ):
        res = self.space.metric.inner_coproduct(
            cotangent_vec_a, cotangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inner_coproduct_vec(self, n_reps, atol):
        # TODO: check if cotangent generatpion makes sense
        base_point = self.data_generator.random_point()
        cotangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        cotangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.inner_coproduct(
            cotangent_vec_a, cotangent_vec_b, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    cotangent_vec_a=cotangent_vec_a,
                    cotangent_vec_b=cotangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["cotangent_vec_a", "cotangent_vec_b", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_squared_norm(self, vector, base_point, expected, atol):
        res = self.space.metric.squared_norm(vector, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_squared_norm_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        vector = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.squared_norm(vector, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(vector=vector, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["vector", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_norm(self, vector, base_point, expected, atol):
        res = self.space.metric.norm(vector, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_norm_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        vector = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.norm(vector, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(vector=vector, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["vector", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_normalize(self, vector, base_point, expected, atol):
        res = self.space.metric.normalize(vector, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_normalize_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        vector = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.normalize(vector, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(vector=vector, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["vector", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_squared_dist(self, point_a, point_b, expected, atol):
        res = self.space.metric.squared_dist(point_a, point_b)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_squared_dist_vec(self, n_reps, atol):
        point_a, point_b = self.data_generator.random_point(2)

        expected = self.space.metric.squared_dist(point_a, point_b)

        vec_data = generate_vectorization_data(
            data=[dict(point_a=point_a, point_b=point_b, expected=expected, atol=atol)],
            arg_names=["point_a", "point_b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_dist(self, point_a, point_b, expected, atol):
        res = self.space.metric.dist(point_a, point_b)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_dist_vec(self, n_reps, atol):
        point_a, point_b = self.data_generator.random_point(2)

        expected = self.space.metric.dist(point_a, point_b)

        vec_data = generate_vectorization_data(
            data=[dict(point_a=point_a, point_b=point_b, expected=expected, atol=atol)],
            arg_names=["point_a", "point_b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_diameter(self, points, expected, atol):
        # TODO: generated automatically. check if correct
        res = self.space.metric.diameter(points)
        self.assertAllClose(res, expected, atol=atol)

    def test_normal_basis(self, basis, base_point, expected, atol):
        res = self.space.metric.normal_basis(basis, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_covariant_riemann_tensor(self, base_point, expected, atol):
        res = self.space.metric.covariant_riemann_tensor(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_covariant_riemann_tensor_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.covariant_riemann_tensor(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_sectional_curvature(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_sectional_curvature_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_scalar_curvature(self, base_point, expected, atol):
        res = self.space.metric.scalar_curvature(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_scalar_curvature_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.scalar_curvature(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)
