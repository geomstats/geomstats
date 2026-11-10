import geomstats.backend as gs
from geomstats.test.assertions import assert_belongs, assert_is_tangent
from geomstats.test.test_case import TestCase


class _ManifoldTestCaseMixins:
    # TODO: check default_coords_type correcteness if intrinsic by comparing
    # with point shape?
    # TODO: need to review to_tangent

    def test_dim(self, expected):
        """Check dimension.

        Parameters
        ----------
        expected : int
            Expected dimension of the manifold.
        """
        self.assertEqual(self.space.dim, expected)

    def test_belongs(self, point, expected, atol=gs.atol):
        """Check membership.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point.
        expected : array-like, shape=[...]
            Boolean evaluating if `point` belongs to the manifold.
        atol : float
            Absolute tolerance.
        """
        res = self.space.belongs(point, atol=atol)
        self.assertAllEqual(res, expected)

    def test_is_tangent(self, vector, base_point, expected, atol=gs.atol):
        """Check if vector is tangent.

        Parameters
        ----------
        vector : array-like, shape=[..., *space.shape]
            Vector.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[...]
            Boolean evaluating if `vector` is tangent at `base_point`.
        atol : float
            Absolute tolerance.
        """
        res = self.space.is_tangent(vector, base_point, atol=atol)
        self.assertAllEqual(res, expected)

    def test_regularize(self, point, expected, atol=gs.atol):
        """Check regularization.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., *space.shape]
            Expected regularized point.
        atol : float
            Absolute tolerance.
        """
        regularized_point = self.space.regularize(point)
        self.assertAllClose(regularized_point, expected, atol=atol)

    def test_belongs_is_true(self, point, atol=gs.atol):
        """Check point belongs to the manifold.

        With random data, this checks random points are generated on the
        manifold.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_belongs(self.space, point, atol=atol)

    def test_assert_is_tangent(self, vector, base_point, atol=gs.atol):
        """Check vector is tangent to the manifold.

        With random data, this checks random tangent vectors are generated
        in the tangent space at `base_point`.

        Parameters
        ----------
        vector : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_is_tangent(self.space, vector, base_point, atol=atol)

    def test_regularize_belongs(self, point, atol=gs.atol):
        """Check regularization maps to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        regularized_point = self.space.regularize(point)

        assert_belongs(self.space, regularized_point, atol=atol)


class ManifoldTestCase(_ManifoldTestCaseMixins, TestCase):
    pass
