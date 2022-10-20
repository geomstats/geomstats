import geomstats.backend as gs
from tests.conftest import Parametrizer
from tests.data.klein_bottle_data import KleinBottleMetricTestData, KleinBottleTestData
from tests.geometry_test_cases import ManifoldTestCase, RiemannianMetricTestCase


class TestKleinBottle(ManifoldTestCase, metaclass=Parametrizer):

    testing_data = KleinBottleTestData()

    skip_test_projection_belongs = True

    def test_equivalent(self, point1, point2, expected):
        space = self.Space()
        is_equivalent = space.equivalent(point1, point2)
        self.assertAllEqual(is_equivalent, expected)

    def test_regularize(self, point, regularized):
        space = self.Space()
        regularized_computed = space.regularize(point)
        self.assertAllClose(regularized_computed, regularized)

    def test_random_point_belongs(self, space_args, n_points, atol):
        """Check that a random point belongs to the manifold.

        Parameters
        ----------
        space_args : tuple
            Arguments to pass to constructor of the manifold.
        n_points : array-like
            Number of random points to sample.
        atol : float
            Absolute tolerance for the belongs function.
        """
        space = self.Space(*space_args)
        random_point = space.random_point(n_points)
        result = space.belongs(random_point, atol=atol)
        self.assertAllEqual(result, gs.array([True] * n_points))


class TestKleinBottleMetric(RiemannianMetricTestCase, metaclass=Parametrizer):

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
        metric = self.Metric(self.testing_data.space)
        distance = metric.dist(point_a, point_b)
        self.assertAllClose(distance, expected)

    def test_diameter(self, points, expected):
        metric = self.Metric(self.testing_data.space)
        diam = metric.diameter(points)
        self.assertAllClose(diam, expected)

    def test_exp(self, base_point, tangent_vec, expected):
        metric = self.Metric(self.testing_data.space)
        exp_result = metric.exp(tangent_vec, base_point)
        self.assertAllClose(exp_result, expected)

    def test_log(self, base_point, point, expected):
        metric = self.Metric(self.testing_data.space)
        log_result = metric.log(point, base_point)
        self.assertAllClose(log_result, expected)
