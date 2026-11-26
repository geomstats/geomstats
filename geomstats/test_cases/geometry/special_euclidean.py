import geomstats.backend as gs
from geomstats.geometry.special_euclidean import (
    SpecialEuclidean,
    homogeneous_representation,
)
from geomstats.test.assertions import assert_belongs
from geomstats.test.test_case import assert_allclose
from geomstats.test_cases.geometry.lie_group import LieGroupTestCase
from geomstats.test_cases.geometry.mixins import ProjectionTestCaseMixins


def homogeneous_representation_test_case(
    rotation,
    translation,
    constant,
    expected,
    atol=gs.atol,
):
    out = homogeneous_representation(rotation, translation, constant)
    assert_allclose(out, expected, atol=atol)


class SpecialEuclideanVectorsTestCase(ProjectionTestCaseMixins, LieGroupTestCase):
    def test_matrix_from_vector(self, vec, expected, atol=gs.atol):
        """Check matrix from vector.

        Parameters
        ----------
        vec : array-like, shape=[..., dim]
            Vector.
        expected : array-like, shape=[..., n + 1, n + 1]
            Expected matrix in homogeneous representation.
        atol : float
            Absolute tolerance.
        """
        mat = self.space.matrix_from_vector(vec)
        self.assertAllClose(mat, expected, atol=atol)

    def test_regularize_tangent_vec(
        self, tangent_vec, base_point, expected, atol=gs.atol
    ):
        """Check tangent vector regularization.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., dim]
            Point.
        expected : array-like, shape=[..., dim]
            Expected regularized tangent vector.
        atol : float
            Absolute tolerance.
        """
        regularized = self.space.regularize_tangent_vec(tangent_vec, base_point)
        self.assertAllClose(regularized, expected, atol=atol)

    def test_regularize_tangent_vec_at_identity(
        self, tangent_vec, expected, atol=gs.atol
    ):
        """Check tangent vector regularization at the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the identity.
        expected : array-like, shape=[..., dim]
            Expected regularized tangent vector.
        atol : float
            Absolute tolerance.
        """
        regularized = self.space.regularize_tangent_vec_at_identity(tangent_vec)
        self.assertAllClose(regularized, expected, atol=atol)

    def test_matrix_from_vector_belongs_to_matrices(self, vec, atol=gs.atol):
        """Check matrix from vector belongs to SE(n) in matrix form.

        Parameters
        ----------
        vec : array-like, shape=[..., dim]
            Vector.
        atol : float
            Absolute tolerance.
        """
        mat = self.space.matrix_from_vector(vec)
        space = SpecialEuclidean(self.space.n, point_type="matrix", equip=False)
        assert_belongs(space, mat, atol=atol)
