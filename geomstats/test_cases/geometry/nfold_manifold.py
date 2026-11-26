import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.mixins import ProjectionTestCaseMixins
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class NFoldManifoldTestCase(ProjectionTestCaseMixins, ManifoldTestCase):
    pass


class NFoldMetricTestCase(RiemannianMetricTestCase):
    pass


class NFoldMetricScalesTestCase(TestCase):
    def setup_method(self):
        if self.space.n_copies != 1:
            raise ValueError("Test case works only for n_copies=1")

    def test_inner_product(
        self, base_point, tangent_vec_a, tangent_vec_b, atol=gs.atol
    ):
        """Check inner product is the scaled inner product of the base manifold.

        Parameters
        ----------
        base_point : array-like, shape=[..., *other_space.shape]
            Point on the base manifold.
        tangent_vec_a : array-like, shape=[..., *other_space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *other_space.shape]
            Tangent vector at `base_point`.
        atol : float
            Absolute tolerance.
        """
        axis = -(self.other_space.point_ndim + 1)
        tangent_vec_a_ = gs.expand_dims(tangent_vec_a, axis=axis)
        tangent_vec_b_ = gs.expand_dims(tangent_vec_b, axis=axis)
        base_point_ = gs.expand_dims(base_point, axis=axis)

        res = self.space.metric.inner_product(
            tangent_vec_a_, tangent_vec_b_, base_point_
        )
        other_res = self.other_space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, other_res, atol=atol)
