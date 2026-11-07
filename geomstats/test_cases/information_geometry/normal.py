import geomstats.backend as gs
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase


def univariate_fisher_rao_dist(point_a, point_b):
    mean_a = point_a[..., 0]
    mean_b = point_b[..., 0]
    sigma_a = point_a[..., 1]
    sigma_b = point_b[..., 1]

    sdiff_mean = (mean_a - mean_b) ** 2
    sdiff_sigma = (sigma_a - sigma_b) ** 2
    ssum_sigma = (sigma_a + sigma_b) ** 2

    return (
        2
        * gs.sqrt(2)
        * gs.arctanh(
            gs.sqrt((sdiff_mean + 2 * sdiff_sigma) / (sdiff_mean + 2 * ssum_sigma))
        )
    )


class UnivariateNormalMetricTestCase(PullbackDiffeoMetricTestCase):
    def test_dist_against_closed_form(self, point_a, point_b, atol=gs.atol):
        """Check distance against its closed form.

        Parameters
        ----------
        point_a : array-like, shape=[..., 2]
            Point on the manifold.
        point_b : array-like, shape=[..., 2]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.dist(point_a, point_b)
        expected = univariate_fisher_rao_dist(point_a, point_b)
        self.assertAllClose(res, expected, atol=atol)

    def test_sectional_curvature_value(
        self, base_point, tangent_vec_a, tangent_vec_b, atol=gs.atol
    ):
        """Check sectional curvature is -1/2.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point on the manifold.
        tangent_vec_a : array-like, shape=[..., 2]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., 2]
            Tangent vector at `base_point`.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertTrue(gs.all(gs.abs(res + 1 / 2) < atol))
