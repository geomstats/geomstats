import geomstats.backend as gs
from geomstats.test.assertions import assert_belongs, assert_is_tangent
from geomstats.test_cases.geometry.complex_manifold import ComplexManifoldTestCase

from .manifold import ManifoldTestCase
from .mixins import ProjectionTestCaseMixins

# TODO: vec with tangent_vecs may not be being tested sufficiently well
# i.e. tests may pass but just because it is the repetition of points

# TODO: define better where to use pytes.mark.mathprop

# TODO: enumerate tests for which random is not enough

# TODO: review uses of gs.ones and gs.zeros

# TODO: review "passes" in tests (maybe not implemented error?)


class _VectorSpaceTestCaseMixins(ProjectionTestCaseMixins):
    def test_random_point_is_tangent(self, point, atol=gs.atol):
        """Check point is tangent.

        In a vector space, points and tangent vectors are identified.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the vector space.
        atol : float
            Absolute tolerance.
        """
        assert_is_tangent(self.space, point, None, atol=atol)

    def test_to_tangent_is_projection(self, point, atol=gs.atol):
        """Check to_tangent is the same as projection.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the vector space.
        atol : float
            Absolute tolerance.
        """
        result = self.space.to_tangent(point)
        expected = self.space.projection(point)

        self.assertAllClose(result, expected, atol=atol)


class VectorSpaceTestCase(_VectorSpaceTestCaseMixins, ManifoldTestCase):
    def test_basis(self, expected, atol=gs.atol):
        """Check basis.

        Parameters
        ----------
        expected : array-like, shape=[dim, *space.shape]
            Expected basis.
        atol : float
            Absolute tolerance.
        """
        self.assertAllClose(self.space.basis, expected, atol=atol)

    def test_basis_cardinality(self):
        """Check number of basis elements is the dimension."""
        basis = self.space.basis
        self.assertEqual(basis.shape[0], self.space.dim)

    def test_basis_belongs(self, atol=gs.atol):
        """Check basis elements belong to the vector space.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        assert_belongs(self.space, self.space.basis, atol=atol)


class ComplexVectorSpaceTestCase(_VectorSpaceTestCaseMixins, ComplexManifoldTestCase):
    pass


class MatrixVectorSpaceTestCase(VectorSpaceTestCase):
    def test_basis_representation(
        self, matrix_representation, expected, atol=gs.atol
    ):
        """Check basis representation.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., *space.shape]
            Point on the vector space.
        expected : array-like, shape=[..., dim]
            Expected coefficients in the basis of the vector space.
        atol : float
            Absolute tolerance.
        """
        vec = self.space.basis_representation(matrix_representation)
        self.assertAllClose(vec, expected, atol=atol)

    def test_matrix_representation(self, basis_representation, expected, atol=gs.atol):
        """Check matrix representation.

        Parameters
        ----------
        basis_representation : array-like, shape=[..., dim]
            Coefficients in the basis of the vector space.
        expected : array-like, shape=[..., *space.shape]
            Expected point on the vector space.
        atol : float
            Absolute tolerance.
        """
        mat = self.space.matrix_representation(basis_representation)
        self.assertAllClose(mat, expected, atol=atol)

    def test_basis_representation_and_basis(
        self, matrix_representation, atol=gs.atol
    ):
        """Check basis representation gives the coefficients in the basis.

        Combining the basis elements with the coefficients recovers the matrix.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., *space.shape]
            Point on the vector space.
        atol : float
            Absolute tolerance.
        """
        vec = self.space.basis_representation(matrix_representation)

        res = gs.einsum("...i,...ijk->...jk", vec, self.space.basis)
        self.assertAllClose(res, matrix_representation, atol=atol)

    def test_matrix_representation_belongs(self, basis_representation, atol=gs.atol):
        """Check matrix representation maps to the vector space.

        Parameters
        ----------
        basis_representation : array-like, shape=[..., dim]
            Coefficients in the basis of the vector space.
        atol : float
            Absolute tolerance.
        """
        point = self.space.matrix_representation(basis_representation)

        assert_belongs(self.space, point, atol=atol)

    def test_matrix_representation_after_basis_representation(
        self, matrix_representation, atol=gs.atol
    ):
        """Check matrix representation is a left inverse of basis representation.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., *space.shape]
            Point on the vector space.
        atol : float
            Absolute tolerance.
        """
        vec = self.space.basis_representation(matrix_representation)
        point = self.space.matrix_representation(vec)

        self.assertAllClose(point, matrix_representation, atol=atol)

    def test_basis_representation_after_matrix_representation(
        self, basis_representation, atol=gs.atol
    ):
        """Check basis representation is a left inverse of matrix representation.

        Parameters
        ----------
        basis_representation : array-like, shape=[..., dim]
            Coefficients in the basis of the vector space.
        atol : float
            Absolute tolerance.
        """
        point = self.space.matrix_representation(basis_representation)
        vec = self.space.basis_representation(point)

        self.assertAllClose(vec, basis_representation, atol=atol)


class ComplexMatrixVectorSpaceTestCase(ComplexVectorSpaceTestCase):
    pass


class LevelSetTestCase(ProjectionTestCaseMixins, ManifoldTestCase):
    # TODO: need to develop `intrinsic_after_extrinsic` and `extrinsic_after_intrinsic`
    # TODO: class to handle `extrinsinc-intrinsic` mixins?

    def test_submersion(self, point, expected, atol=gs.atol):
        """Check submersion.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the embedding space.
        expected : array-like, shape=[..., *submersion_shape]
            Expected submersed point.
        atol : float
            Absolute tolerance.
        """
        submersed_point = self.space.submersion(point)
        self.assertAllClose(submersed_point, expected, atol=atol)

    def test_tangent_submersion(self, vector, point, expected, atol=gs.atol):
        """Check tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., *space.shape]
            Tangent vector to the embedding space at `point`.
        point : array-like, shape=[..., *space.shape]
            Point on the embedding space.
        expected : array-like, shape=[..., *tangent_submersion_shape]
            Expected submersed vector.
        atol : float
            Absolute tolerance.
        """
        submersed_vector = self.space.tangent_submersion(vector, point)
        self.assertAllClose(submersed_vector, expected, atol=atol)

    def test_submersion_is_zero(self, point, atol=gs.atol):
        """Check submersion vanishes.

        The submersion vanishes exactly on the level set, so `point`
        must belong to it.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        submersed_point = self.space.submersion(point)
        self.assertAllClose(
            submersed_point, gs.zeros_like(submersed_point), atol=atol
        )

    def test_tangent_submersion_is_zero(self, vector, point, atol=gs.atol):
        """Check tangent submersion vanishes.

        The tangent submersion vanishes exactly on tangent vectors to the
        level set, so `vector` must be tangent to it at `point`.

        Parameters
        ----------
        vector : array-like, shape=[..., *space.shape]
            Tangent vector to the manifold at `point`.
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        submersed_vector = self.space.tangent_submersion(vector, point)
        self.assertAllClose(
            submersed_vector, gs.zeros_like(submersed_vector), atol=atol
        )


class _OpenSetTestCaseMixins(ProjectionTestCaseMixins):
    def test_to_tangent_is_tangent_in_embedding_space(
        self, tangent_vec, base_point, atol=gs.atol
    ):
        """Check tangent vector is tangent to the embedding space.

        The tangent space of an open set coincides with the one of its
        embedding space.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_is_tangent(
            self.space.embedding_space, tangent_vec, base_point, atol=atol
        )


class OpenSetTestCase(_OpenSetTestCaseMixins, ManifoldTestCase):
    pass


class VectorSpaceOpenSetTestCase(OpenSetTestCase):
    pass


class ComplexVectorSpaceOpenSetTestCase(
    _OpenSetTestCaseMixins, ComplexManifoldTestCase
):
    pass


class ImmersedSetTestCase(ProjectionTestCaseMixins, ManifoldTestCase):
    def test_immersion(self, point, expected, atol=gs.atol):
        """Check immersion.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., *embedding_space.shape]
            Expected image of `point`.
        atol : float
            Absolute tolerance.
        """
        res = self.space.immersion(point)
        self.assertAllClose(res, expected, atol=atol)

    def test_tangent_immersion(self, tangent_vec, base_point, expected, atol=gs.atol):
        """Check tangent immersion.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., *embedding_space.shape]
            Expected image of `tangent_vec`.
        atol : float
            Absolute tolerance.
        """
        res = self.space.tangent_immersion(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_jacobian_immersion(self, base_point, expected, atol=gs.atol):
        """Check Jacobian of the immersion.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., embedding_dim, dim]
            Expected Jacobian.
        atol : float
            Absolute tolerance.
        """
        res = self.space.jacobian_immersion(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_hessian_immersion(self, base_point, expected, atol=gs.atol):
        """Check Hessian of the immersion.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., embedding_dim, dim, dim]
            Expected Hessian.
        atol : float
            Absolute tolerance.
        """
        res = self.space.hessian_immersion(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_immersion_belongs(self, point, atol=gs.atol):
        """Check immersion belongs to the embedding space.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        embedded_point = self.space.immersion(point)

        assert_belongs(self.space.embedding_space, embedded_point, atol=atol)

    def test_tangent_immersion_is_tangent(self, tangent_vec, base_point, atol=gs.atol):
        """Check tangent immersion is tangent to the embedding space.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        embedded_point = self.space.immersion(base_point)
        embedded_tangent_vec = self.space.tangent_immersion(tangent_vec, base_point)

        assert_is_tangent(
            self.space.embedding_space,
            embedded_tangent_vec,
            embedded_point,
            atol=atol,
        )


class DiffeomorphicManifoldTestCase(ManifoldTestCase):
    pass
