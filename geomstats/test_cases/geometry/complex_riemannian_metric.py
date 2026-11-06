import geomstats.backend as gs
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class ComplexRiemannianMetricTestCase(RiemannianMetricTestCase):
    def test_inner_product_is_complex(self, tangent_vec_a, tangent_vec_b, base_point):
        """Check inner product is complex.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        """
        res = self.space.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        self.assertTrue(gs.is_complex(res))

    def test_dist_is_real(self, point_a, point_b):
        """Check distance is real.

        Parameters
        ----------
        point_a : array-like, shape=[..., *space.shape]
            Point on the manifold.
        point_b : array-like, shape=[..., *space.shape]
            Point on the manifold.
        """
        res = self.space.metric.dist(point_a, point_b)
        self.assertTrue(not gs.is_complex(res))

    def test_log_is_complex(self, point, base_point):
        """Check logarithm is complex.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        """
        res = self.space.metric.log(point, base_point)
        self.assertTrue(gs.is_complex(res))

    def test_exp_is_complex(self, tangent_vec, base_point):
        """Check exponential is complex.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        """
        res = self.space.metric.exp(tangent_vec, base_point)
        self.assertTrue(gs.is_complex(res))
