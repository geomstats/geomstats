import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.manifold import _ManifoldTestCaseMixins


class ComplexManifoldTestCase(_ManifoldTestCaseMixins, TestCase):
    def test_random_point_is_complex(self, point):
        """Check random points are complex.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Random point on the manifold.
        """
        self.assertTrue(gs.is_complex(point))

    def test_random_point_imaginary_nonzero(self, point):
        """Check random points have a nonzero imaginary part.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Random point on the manifold.
        """
        res = gs.imag(point)
        self.assertTrue(gs.sum(gs.abs(res)) > 0.0)

    def test_random_tangent_vec_is_complex(self, tangent_vec, base_point):
        """Check random tangent vectors are complex.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Random tangent vector at base point.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        """
        self.assertTrue(gs.is_complex(tangent_vec))

    def test_random_tangent_vec_imaginary_nonzero(self, tangent_vec, base_point):
        """Check random tangent vectors have a nonzero imaginary part.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Random tangent vector at base point.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        """
        res = gs.imag(tangent_vec)
        self.assertTrue(gs.sum(gs.abs(res)) > 0.0)
