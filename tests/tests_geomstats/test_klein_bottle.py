"""Unit tests for Klein bottle manifold."""

import geomstats._backend as gs
from tests.conftest import Parametrizer
from tests.data.klein_bottle_data import (KleinBottleMetricTestData,
                                          KleinBottleTestData)
from tests.geometry_test_cases import (ManifoldTestCase,
                                       RiemannianMetricTestCase)


class TestKleinBottle(ManifoldTestCase, metaclass=Parametrizer):
    """Class to test Klein bottle manifold."""

    testing_data = KleinBottleTestData()

    skip_test_projection_belongs = True

    def test_equivalent(self, point_a, point_b, expected):
        """Test equivalency given two points."""
        space = self.Space()
        is_equivalent = space.equivalent(point_a, point_b)
        self.assertAllEqual(is_equivalent, expected)

    def test_regularize(self, point, regularized):
        """Test the regularize method."""
        space = self.Space()
        regularized_computed = space.regularize(point)
        self.assertAllClose(regularized_computed, regularized)

    def test_regularize_correct_domain(self, points):
        """Test regularization between 0 and 1."""
        space = self.Space()
        regularized_computed = space.regularize(points)
        greater_zero = gs.all(regularized_computed >= 0)
        smaller_one = gs.all(regularized_computed < 1)
        self.assertTrue(greater_zero and smaller_one)

    def test_not_belongs(self, point, expected):
        """Test the belongs method."""
        space = self.Space()
        belongs_result = space.belongs(point)
        self.assertAllEqual(belongs_result, expected)


class TestKleinBottleMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    """Class to test Klein bottle metric."""

    testing_data = KleinBottleMetricTestData()

    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_sectional_curvature_shape = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_riemann_tensor_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_scalar_curvature_shape = True

    def test_dist(self, point_a, point_b, expected):
        """Test the distance between two points."""
        metric = self.Metric(self.testing_data.space)
        distance = metric.dist(point_a, point_b)
        self.assertAllClose(distance, expected)

    def test_diameter(self, points, expected):
        """Test the diameter given two points."""
        metric = self.Metric(self.testing_data.space)
        diam = metric.diameter(points)
        self.assertAllClose(diam, expected)

    def test_exp(self, base_point, tangent_vec, expected):
        """Test the exp method."""
        metric = self.Metric(self.testing_data.space)
        exp_result = metric.exp(tangent_vec, base_point)
        self.assertAllClose(exp_result, expected)

    def test_log(self, base_point, point, expected):
        """Test the log method."""
        metric = self.Metric(self.testing_data.space)
        log_result = metric.log(point, base_point)
        self.assertAllClose(log_result, expected)
