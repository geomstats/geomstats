import pytest

import geomstats.backend as gs
from geomstats.test.random import (
    EmbeddedSpaceRandomDataGenerator,
    MatrixVectorSpaceRandomDataGenerator,
    VectorSpaceRandomDataGenerator,
)
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.complex_manifold import ComplexManifoldTestCase
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.mixins import ProjectionTestCaseMixins
from geomstats.vectorization import get_batch_shape

# TODO: vec with tangent_vecs may not be being tested sufficiently well
# i.e. tests may pass but just because it is the repetition of points

# TODO: define better where to use pytes.mark.mathprop

# TODO: enumerate tests for which random is not enough

# TODO: review uses of gs.ones and gs.zeros

# TODO: review "passes" in tests (maybe not implemented error?)


class _VectorSpaceTestCaseMixins(ProjectionTestCaseMixins):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = VectorSpaceRandomDataGenerator(self.space)
        super().setup_method()

    @pytest.mark.random
    def test_random_point_is_tangent(self, n_points, atol):
        """Check random point is tangent.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        points = self.data_generator.random_point(n_points)

        res = self.space.is_tangent(points, atol=atol)
        self.assertAllEqual(res, gs.ones(n_points, dtype=bool))

    @pytest.mark.random
    def test_to_tangent_is_projection(self, n_points, atol):
        """Check to_tangent is same as projection.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        vec = self.data_generator.random_point(n_points)
        result = self.space.to_tangent(vec)
        expected = self.space.projection(vec)

        self.assertAllClose(result, expected, atol=atol)

    @pytest.mark.mathprop
    def test_basis_cardinality(self):
        """Check number of basis elements is the dimension.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        basis = self.space.basis
        self.assertEqual(basis.shape[0], self.space.dim)

    @pytest.mark.mathprop
    def test_basis_belongs(self, atol):
        """Check basis elements belong to vector space.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        result = self.space.belongs(self.space.basis, atol=atol)
        self.assertAllEqual(result, gs.ones_like(result))

    def test_basis(self, expected, atol):
        self.assertAllClose(self.space.basis, expected, atol=atol)


class VectorSpaceTestCase(_VectorSpaceTestCaseMixins, ManifoldTestCase):
    pass


class ComplexVectorSpaceTestCase(_VectorSpaceTestCaseMixins, ComplexManifoldTestCase):
    pass


class _MatrixVectorSpaceTestCaseMixins:
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = MatrixVectorSpaceRandomDataGenerator(self.space)
        super().setup_method()

    def test_basis_representation(self, matrix_representation, expected, atol):
        vec = self.space.basis_representation(matrix_representation)
        self.assertAllClose(vec, expected, atol=atol)

    @pytest.mark.vec
    def test_basis_representation_vec(self, n_reps, atol):
        matrix_representation = self.data_generator.random_point()
        expected = self.space.basis_representation(matrix_representation)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    matrix_representation=matrix_representation,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["matrix_representation"],
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

    def test_matrix_representation(self, basis_representation, expected, atol):
        mat = self.space.matrix_representation(basis_representation)
        self.assertAllClose(mat, expected, atol=atol)

    @pytest.mark.vec
    def test_matrix_representation_vec(self, n_reps, atol):
        basis_representation = gs.random.rand(self.space.dim)
        expected = self.space.matrix_representation(basis_representation)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    basis_representation=basis_representation,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["basis_representation"],
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


class MatrixVectorSpaceTestCase(_MatrixVectorSpaceTestCaseMixins, VectorSpaceTestCase):
    pass


class ComplexMatrixVectorSpaceTestCase(
    _MatrixVectorSpaceTestCaseMixins, ComplexVectorSpaceTestCase
):
    pass


class LevelSetTestCase(ProjectionTestCaseMixins, ManifoldTestCase):
    # TODO: need to develop `intrinsic_after_extrinsic` and `extrinsic_after_intrinsic`
    # TODO: class to handle `extrinsinc-intrinsic` mixins?

    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = EmbeddedSpaceRandomDataGenerator(self.space)
        super().setup_method()

    def test_submersion(self, point, expected, atol):
        submersed_point = self.space.submersion(point)
        self.assertAllClose(submersed_point, expected, atol=atol)

    def test_submersion_is_zero(self, point, submersion_shape, atol):
        # TODO: keep?
        batch_shape = get_batch_shape(self.space.point_ndim, point)
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
        batch_shape = get_batch_shape(self.space.point_ndim, tangent_vector, point)
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
            vectorization_type="sym" if self.tangent_to_multiple else "repetition-0",
            expected_name="expected",
        )
        self._test_vectorization(vec_data)


class _OpenSetTestCaseMixins(ProjectionTestCaseMixins):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = EmbeddedSpaceRandomDataGenerator(self.space)
        super().setup_method()

    @pytest.mark.random
    def test_to_tangent_is_tangent_in_embedding_space(self, n_points):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        res = self.space.embedding_space.is_tangent(tangent_vec, base_point)

        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)


class OpenSetTestCase(_OpenSetTestCaseMixins, ManifoldTestCase):
    pass


class VectorSpaceOpenSetTestCase(OpenSetTestCase):
    pass


class ComplexVectorSpaceOpenSetTestCase(
    _OpenSetTestCaseMixins, ComplexManifoldTestCase
):
    pass


class ImmersedSetTestCase(ProjectionTestCaseMixins, ManifoldTestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = EmbeddedSpaceRandomDataGenerator(self.space)
        super().setup_method()

    def test_immersion(self, point, expected, atol):
        res = self.space.immersion(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_immersion_belongs(self, n_points, atol):
        point = self.data_generator.random_point(n_points)

        embedded_point = self.space.immersion(point)

        expected = gs.ones(n_points, dtype=bool)
        res = self.space.embedding_space.belongs(embedded_point, atol=atol)
        self.assertAllEqual(res, expected)

    def test_tangent_immersion(self, tangent_vec, base_point, expected, atol):
        res = self.space.tangent_immersion(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_tangent_immersion_is_tangent(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        embedded_point = self.space.immersion(base_point)
        embedded_tangent_vec = self.space.tangent_immersion(tangent_vec, base_point)

        expected = gs.ones(n_points, dtype=bool)
        res = self.space.embedding_space.is_tangent(
            embedded_tangent_vec, embedded_point, atol=atol
        )
        self.assertAllEqual(res, expected)

    def test_jacobian_immersion(self, base_point, expected, atol):
        res = self.space.jacobian_immersion(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_hessian_immersion(self, base_point, expected, atol):
        res = self.space.hessian_immersion(base_point)
        self.assertAllClose(res, expected, atol=atol)


class DiffeomorphicManifoldTestCase(ManifoldTestCase):
    pass
