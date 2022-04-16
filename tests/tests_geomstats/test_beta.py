"""Unit tests for the beta manifold."""


from scipy.stats import beta

import geomstats.backend as gs
import geomstats.tests
from geomstats.information_geometry.beta import BetaDistributions, BetaMetric
from tests.conftest import Parametrizer
from tests.data.beta_data import BetaDistributionsTestsData, BetaMetricTestData
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase


class TestBetaDistributions(OpenSetTestCase, metaclass=Parametrizer):
    space = BetaDistributions

    testing_data = BetaDistributionsTestsData()

    def test_point_to_pdf(self, x):
        point = BetaDistributions().random_point()
        pdf = BetaDistributions().point_to_pdf(point)
        result = pdf(x)
        expected = beta.pdf(x, a=point[0], b=point[1])
        self.assertAllClose(result, expected)

    def test_point_to_pdf_vectorization(self, x):
        point = BetaDistributions().random_point(n_samples=2)
        pdf = BetaDistributions().point_to_pdf(point)
        result = pdf(x)
        pdf1 = beta.pdf(x, a=point[0, 0], b=point[0, 1])
        pdf2 = beta.pdf(x, a=point[1, 0], b=point[1, 1])
        expected = gs.stack([gs.array(pdf1), gs.array(pdf2)], axis=1)
        self.assertAllClose(result, expected)


class TestBetaMetric(RiemannianMetricTestCase, metaclass=Parametrizer):

    space = BetaMetric
    connection = metric = BetaMetric
    skip_test_exp_shape = True  # because several base points for one vector
    skip_test_log_shape = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_exp_belongs = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_log_is_tangent = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_dist_is_symmetric = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_dist_is_positive = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_squared_dist_is_symmetric = True
    skip_test_squared_dist_is_positive = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_dist_is_norm_of_log = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_dist_point_to_itself_is_zero = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_log_after_exp = True
    skip_test_exp_after_log = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_ladder_parallel_transport = True

    testing_data = BetaMetricTestData()

    def test_metric_matrix(self, point, expected):
        result = self.metric().metric_matrix(point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_autograd_only
    def test_exp(self, n_samples):
        """Test Exp.

        Test that the Riemannian exponential at points on the first
        bisector computed in the direction of the first bisector stays
        on the first bisector.
        """
        points = BetaDistributions().random_point(n_samples)
        vectors = BetaDistributions().random_point(n_samples)
        initial_vectors = gs.array([[vec_x, vec_x] for vec_x in vectors[:, 0]])
        points = gs.array([[param_a, param_a] for param_a in points[:, 0]])
        result_points = self.metric().exp(initial_vectors, points)
        result = gs.isclose(result_points[:, 0], result_points[:, 1]).all()
        expected = gs.array([True] * n_samples)
        self.assertAllClose(expected, result)

    def test_christoffels_shape(self, n_samples):
        """Test Christoffel synbols.
        Check vectorization of Christoffel symbols.
        """
        points = BetaDistributions().random_point(n_samples)
        dim = BetaDistributions().dim
        christoffel = self.metric().christoffels(points)
        result = christoffel.shape
        expected = gs.array([n_samples, dim, dim, dim])
        self.assertAllClose(result, expected)
