import geomstats.backend as gs
from geomstats.test.assertions import assert_belongs, assert_is_tangent
from geomstats.test.test_case import TestCase


class RetractionTestCase(TestCase):
    def test_retraction(self, tangent_vec, base_point, expected, atol=gs.atol):
        """Check retraction.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., *space.shape]
            Expected point on the manifold.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.retraction(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_lifting(self, point, base_point, expected, atol=gs.atol):
        """Check lifting.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., *space.shape]
            Expected tangent vector at `base_point`.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.lifting(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_retraction_belongs(self, tangent_vec, base_point, atol=gs.atol):
        """Check retraction gives a point on the manifold.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        point = self.space.metric.retraction(tangent_vec, base_point)
        assert_belongs(self.space, point, atol=atol)

    def test_lifting_is_tangent(self, point, base_point, atol=gs.atol):
        """Check lifting gives a tangent vector.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        tangent_vec = self.space.metric.lifting(point, base_point)

        assert_is_tangent(self.space, tangent_vec, base_point, atol=atol)

    def test_lifting_after_retraction(self, tangent_vec, base_point, atol=gs.atol):
        """Check lifting is a left inverse of the retraction.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        point = self.space.metric.retraction(tangent_vec, base_point)
        tangent_vec_ = self.space.metric.lifting(point, base_point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    def test_retraction_after_lifting(self, point, base_point, atol=gs.atol):
        """Check retraction is a left inverse of the lifting.

        Lifting is a local inverse of the retraction, so `point` must be
        close to `base_point`.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        tangent_vec = self.space.metric.lifting(point, base_point)

        point_ = self.space.metric.retraction(tangent_vec, base_point)
        self.assertAllClose(point_, point, atol=atol)
