import geomstats.backend as gs
from geomstats.test_cases.information_geometry.dirichlet import (
    DirichletDistributionsTestCase,
    DirichletMetricTestCase,
)


class BetaDistributionsTestCase(DirichletDistributionsTestCase):
    def _check_sample_belongs_to_support(self, sample, atol):
        self.assertTrue(gs.all(sample >= 0))
        self.assertTrue(gs.all(sample <= 1))


class BetaMetricTestCase(DirichletMetricTestCase):
    def test_metric_det(self, point, expected, atol=gs.atol):
        """Check metric determinant.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point on the manifold.
        expected : array-like, shape=[...]
            Expected determinant.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.metric_det(point)
        self.assertAllClose(res, expected, atol=atol)

    def test_metric_det_against_metric_matrix(self, point, atol=gs.atol):
        """Check metric determinant against the metric matrix.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.metric_det(point)
        expected = gs.linalg.det(self.space.metric.metric_matrix(point))
        self.assertAllClose(res, expected, atol=atol)

    def test_metric_det_lower_bound(self, base_point):
        """Check metric determinant lower bound.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point on the manifold.

        References
        ----------
        .. [BP2019] Brigant, A.L. and Puechmorel, S. (2019)
        ‘The Fisher-Rao geometry of beta distributions applied to the study of
        canonical moments’. arXiv.
        Available at: https://doi.org/10.48550/arXiv.1904.08247.
        """
        res = self.space.metric.metric_det(base_point)

        alpha = base_point[..., 0]
        beta = base_point[..., 1]

        num = 1 + alpha + beta
        den = 2 * alpha * beta * gs.power((alpha + beta), 2)
        lower_bound = num / den
        self.assertTrue(gs.all(res > lower_bound))

    def test_sectional_curvature_lower_bound(
        self, base_point, tangent_vec_a, tangent_vec_b
    ):
        """Check sectional curvature lower bound.

        It is still a conjecture (see [BPP2020]_).

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point on the manifold.
        tangent_vec_a : array-like, shape=[..., 2]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., 2]
            Tangent vector at `base_point`.

        References
        ----------
        .. [BPP2020] Brigant, A.L., Preston, S. and Puechmorel, S. (2021)
        ‘Fisher-Rao geometry of Dirichlet distributions’, Differential Geometry
        and its Applications, 74, p. 101702.
        Available at: https://doi.org/10.1016/j.difgeo.2020.101702.
        """
        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        lower_bound = -1 / 2
        self.assertTrue(gs.all(res > lower_bound))
