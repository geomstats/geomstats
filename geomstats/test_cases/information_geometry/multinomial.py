import geomstats.backend as gs
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)


def fisher_rao_dist(point_a, point_b, n_draws):
    return 2 * gs.sqrt(n_draws) * gs.arccos(gs.sum(gs.sqrt(point_a * point_b), axis=-1))


class MultinomialDistributionsTestCase(
    InformationManifoldMixinTestCase, LevelSetTestCase
):
    def _check_sample_belongs_to_support(self, sample, _atol):
        self.assertTrue(gs.all(sample >= 0.0))
        self.assertTrue(gs.all(gs.sum(sample, axis=-1) == self.space.n_draws))


class MultinomialMetricTestCase(RiemannianMetricTestCase):
    def test_sectional_curvature_is_positive(
        self, base_point, tangent_vec_a, tangent_vec_b
    ):
        """Check sectional curvature is positive.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim + 1]
            Point on the manifold.
        tangent_vec_a : array-like, shape=[..., dim + 1]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Tangent vector at `base_point`.
        """
        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertTrue(gs.all(res > 0))

    def test_dist_against_closed_form(self, point_a, point_b, atol=gs.atol):
        """Check distance against its closed form.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim + 1]
            Point on the manifold.
        point_b : array-like, shape=[..., dim + 1]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.dist(point_a, point_b)
        expected = fisher_rao_dist(point_a, point_b, self.space.n_draws)
        self.assertAllClose(res, expected, atol=atol)
