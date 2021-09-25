"""Unit tests for the Riemannian metrics."""

import warnings

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric


class TestRiemannianMetric(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=UserWarning)
        gs.random.seed(0)
        self.dim = 2
        self.euc = Euclidean(dim=self.dim)
        self.sphere = Hypersphere(dim=self.dim)
        self.euc_metric = EuclideanMetric(dim=self.dim)
        self.sphere_metric = HypersphereMetric(dim=self.dim)

        def _euc_metric_matrix(base_point):
            """Return matrix of Euclidean inner-product."""
            dim = base_point.shape[-1]
            return gs.eye(dim)

        def _sphere_metric_matrix(base_point):
            """Return sphere's metric in spherical coordinates."""
            theta = base_point[..., 0]
            mat = gs.array([[1.0, 0.0], [0.0, gs.sin(theta) ** 2]])
            return mat

        new_euc_metric = RiemannianMetric(dim=self.dim)
        new_euc_metric.metric_matrix = _euc_metric_matrix

        new_sphere_metric = RiemannianMetric(dim=self.dim)
        new_sphere_metric.metric_matrix = _sphere_metric_matrix

        self.new_euc_metric = new_euc_metric
        self.new_sphere_metric = new_sphere_metric

    def test_cometric_matrix(self):
        base_point = self.euc.random_point()

        result = self.euc_metric.metric_inverse_matrix(base_point)
        expected = gs.eye(self.dim)

        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_and_torch_only
    def test_metric_derivative_euc_metric(self):
        base_point = self.euc.random_point()

        result = self.euc_metric.inner_product_derivative_matrix(base_point)
        expected = gs.zeros((self.dim,) * 3)

        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_and_torch_only
    def test_metric_derivative_new_euc_metric(self):
        base_point = self.euc.random_point()

        result = self.new_euc_metric.inner_product_derivative_matrix(base_point)
        expected = gs.zeros((self.dim,) * 3)

        self.assertAllClose(result, expected)

    def test_inner_product_new_euc_metric(self):
        base_point = self.euc.random_point()
        tan_a = self.euc.random_point()
        tan_b = self.euc.random_point()
        expected = gs.dot(tan_a, tan_b)

        result = self.new_euc_metric.inner_product(tan_a, tan_b, base_point=base_point)

        self.assertAllClose(result, expected)

    def test_inner_product_new_sphere_metric(self):
        base_point = gs.array([gs.pi / 3.0, gs.pi / 5.0])
        tan_a = gs.array([0.3, 0.4])
        tan_b = gs.array([0.1, -0.5])
        expected = -0.12

        result = self.new_sphere_metric.inner_product(
            tan_a, tan_b, base_point=base_point
        )

        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_and_torch_only
    def test_christoffels_eucl_metric(self):
        base_point = self.euc.random_point()

        result = self.euc_metric.christoffels(base_point)
        expected = gs.zeros((self.dim,) * 3)

        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_and_torch_only
    def test_christoffels_new_eucl_metric(self):
        base_point = self.euc.random_point()

        result = self.new_euc_metric.christoffels(base_point)
        expected = gs.zeros((self.dim,) * 3)

        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_christoffels_sphere_metrics(self):
        base_point = gs.array([gs.pi / 10.0, gs.pi / 9.0])

        expected = self.sphere_metric.christoffels(base_point)
        result = self.new_sphere_metric.christoffels(base_point)

        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_and_torch_only
    def test_exp_new_eucl_metric(self):
        base_point = self.euc.random_point()
        tan = self.euc.random_point()

        expected = base_point + tan
        result = self.new_euc_metric.exp(tan, base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_and_torch_only
    def test_log_new_eucl_metric(self):
        base_point = self.euc.random_point()
        point = self.euc.random_point()

        expected = point - base_point
        result = self.new_euc_metric.log(point, base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_exp_new_sphere_metric(self):
        base_point = gs.array([gs.pi / 10.0, gs.pi / 9.0])
        tan = gs.array([gs.pi / 2.0, 0.0])

        expected = gs.array([gs.pi / 10.0 + gs.pi / 2.0, gs.pi / 9.0])
        result = self.new_sphere_metric.exp(tan, base_point)
        self.assertAllClose(result, expected)
