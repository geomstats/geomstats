import geomstats.backend as gs
from geomstats.geometry.base import ImmersedSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.invariant_metric import BiInvariantMetric
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.test_cases.geometry.diffeo import CircleSO2Diffeo
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class CircleAsSO2Metric(PullbackDiffeoMetric):
    def __init__(self, space, image_space):
        if not space.dim == 1:
            raise ValueError(
                "This dummy class using SO(2) metric for S1 has "
                "a meaning only when dim=1"
            )
        if not isinstance(image_space.metric, BiInvariantMetric):
            raise ValueError("Image space must be equipped with a bi-invariant metric")

        diffeo = CircleSO2Diffeo()
        super().__init__(space=space, diffeo=diffeo, image_space=image_space)


class CircleIntrinsic(ImmersedSet):
    def __init__(self, equip=True):
        super().__init__(dim=1, equip=equip)

    def immersion(self, point):
        return gs.hstack([gs.cos(point), gs.sin(point)])

    def _define_embedding_space(self):
        return Euclidean(dim=self.dim + 1)


class SphereIntrinsic(ImmersedSet):
    def __init__(self, equip=True):
        super().__init__(dim=2, equip=equip)

    def immersion(self, point):
        theta = point[..., 0]
        phi = point[..., 1]
        return gs.stack(
            [
                gs.cos(phi) * gs.sin(theta),
                gs.sin(phi) * gs.sin(theta),
                gs.cos(theta),
            ],
            axis=-1,
        )

    def _define_embedding_space(self):
        return Euclidean(dim=self.dim + 1)


class PullbackMetricTestCase(RiemannianMetricTestCase):
    def test_second_fundamental_form(self, base_point, expected, atol):
        res = self.space.metric.second_fundamental_form(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_mean_curvature_vector(self, base_point, expected, atol):
        res = self.space.metric.mean_curvature_vector(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_mean_curvature_vector_norm(self, base_point, expected, atol):
        mean_curvature = self.space.metric.mean_curvature_vector(base_point)
        res = gs.linalg.norm(mean_curvature)
        self.assertAllClose(res, expected, atol=atol)


class PullbackDiffeoMetricTestCase(RiemannianMetricTestCase):
    pass
