import pytest

import geomstats.backend as gs
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.test_cases.information_geometry.dirichlet import (
    DirichletDistributionsTestCase,
    DirichletMetricTestCase,
)

#  TODO: add againts numpy test case for pdf?


def sectional_curvature(tangent_vec_a, tangent_vec_b, base_point):
    x, y = base_point[..., 0], base_point[..., 1]
    detg = gs.polygamma(1, x) * gs.polygamma(1, y) - gs.polygamma(1, x + y) * (
        gs.polygamma(1, x) + gs.polygamma(1, y)
    )
    return (
        gs.polygamma(2, x)
        * gs.polygamma(2, y)
        * gs.polygamma(2, x + y)
        * (
            gs.polygamma(1, x) / gs.polygamma(2, x)
            + gs.polygamma(1, y) / gs.polygamma(2, y)
            - gs.polygamma(1, x + y) / gs.polygamma(2, x + y)
        )
        / (4 * detg**2)
    )


def metric_matrix(base_point):
    param_a = base_point[..., 0]
    param_b = base_point[..., 1]
    vector = gs.stack(
        [
            gs.polygamma(1, param_a) - gs.polygamma(1, param_a + param_b),
            -gs.polygamma(1, param_a + param_b),
            gs.polygamma(1, param_b) - gs.polygamma(1, param_a + param_b),
        ],
        axis=-1,
    )
    return SymmetricMatrices.matrix_representation(vector)


def christoffels(base_point):
    def coefficients(param_a, param_b):
        """Christoffel coefficients for the beta distributions."""
        poly1a = gs.polygamma(1, param_a)
        poly2a = gs.polygamma(2, param_a)
        poly1b = gs.polygamma(1, param_b)
        poly2b = gs.polygamma(2, param_b)
        poly1ab = gs.polygamma(1, param_a + param_b)
        poly2ab = gs.polygamma(2, param_a + param_b)
        metric_det = 2 * (poly1a * poly1b - poly1ab * (poly1a + poly1b))

        c1 = (poly2a * (poly1b - poly1ab) - poly1b * poly2ab) / metric_det
        c2 = -poly1b * poly2ab / metric_det
        c3 = (poly2b * poly1ab - poly1b * poly2ab) / metric_det
        return c1, c2, c3

    param_a, param_b = base_point[..., 0], base_point[..., 1]
    c1, c2, c3 = coefficients(param_a, param_b)
    c4, c5, c6 = coefficients(param_b, param_a)
    vector_0 = gs.stack([c1, c2, c3], axis=-1)
    vector_1 = gs.stack([c6, c5, c4], axis=-1)
    gamma_0 = SymmetricMatrices.matrix_representation(vector_0)
    gamma_1 = SymmetricMatrices.matrix_representation(vector_1)

    return gs.stack([gamma_0, gamma_1], axis=-3)


class BetaDistributionsTestCase(DirichletDistributionsTestCase):
    def _check_sample_belongs_to_support(self, sample, atol):
        self.assertTrue(gs.all(sample >= 0))
        self.assertTrue(gs.all(sample <= 1))


class BetaMetricTestCase(DirichletMetricTestCase):
    def test_metric_det(self, point, expected, atol):
        res = self.space.metric.metric_det(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_metric_det_against_metric_matrix(self, n_points, atol):
        point = self.data_generator.random_point(n_points)

        res = self.space.metric.metric_det(point)
        expected = gs.linalg.det(self.space.metric.metric_matrix(point))
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_metric_det_lower_bound(self, n_points):
        """Check metric determinant lower bound.

        References
        ----------
        .. [BP2019] Brigant, A.L. and Puechmorel, S. (2019)
        ‘The Fisher-Rao geometry of beta distributions applied to the study of
        canonical moments’. arXiv.
        Available at: https://doi.org/10.48550/arXiv.1904.08247.
        """
        base_point = self.data_generator.random_point(n_points)
        res = self.space.metric.metric_det(base_point)

        alpha = base_point[..., 0]
        beta = base_point[..., 1]

        num = 1 + alpha + beta
        den = 2 * alpha * beta * gs.power((alpha + beta), 2)
        lower_bound = num / den
        self.assertTrue(gs.all(res > lower_bound))

    @pytest.mark.random
    def test_sectional_curvature_against_closed_form(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        expected = sectional_curvature(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_sectional_curvature_lower_bound(self, n_points):
        """Check sectional curvature lower bound.

        It is still a conjecture (see [BPP2020]_).

        References
        ----------
        .. [BPP2020] Brigant, A.L., Preston, S. and Puechmorel, S. (2021)
        ‘Fisher-Rao geometry of Dirichlet distributions’, Differential Geometry
        and its Applications, 74, p. 101702.
        Available at: https://doi.org/10.1016/j.difgeo.2020.101702.
        """
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        lower_bound = -1 / 2
        self.assertTrue(gs.all(res > lower_bound))

    @pytest.mark.random
    def test_metric_matrix_against_closed_form(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.metric_matrix(base_point)
        expected = metric_matrix(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_christoffels_against_closed_form(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.christoffels(base_point)
        expected = christoffels(base_point)
        self.assertAllClose(res, expected, atol=atol)
