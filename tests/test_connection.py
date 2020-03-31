"""Unit tests for the affine connections."""

import warnings

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.connection import Connection
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.hypersphere import Hypersphere


class TestConnectionMethods(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)

        self.dimension = 4
        self.euc_metric = EuclideanMetric(dimension=self.dimension)

        self.connection = Connection(dimension=2)
        self.hypersphere = Hypersphere(dimension=2)

    def test_metric_matrix(self):
        base_point = gs.array([0., 1., 0., 0.])

        result = self.euc_metric.inner_product_matrix(base_point)
        expected = gs.array([gs.eye(self.dimension)])

        with self.session():
            self.assertAllClose(result, expected)

    def test_cometric_matrix(self):
        base_point = gs.array([0., 1., 0., 0.])

        result = self.euc_metric.inner_product_inverse_matrix(base_point)
        expected = gs.array([gs.eye(self.dimension)])

        with self.session():
            self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_metric_derivative(self):
        base_point = gs.array([0., 1., 0., 0.])

        result = self.euc_metric.inner_product_derivative_matrix(base_point)
        expected = gs.zeros((1,) + (self.dimension, ) * 3)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_christoffels(self):
        base_point = gs.array([0., 1., 0., 0.])

        result = self.euc_metric.christoffels(base_point)
        expected = gs.zeros((1,) + (self.dimension, ) * 3)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_parallel_transport(self):
        n_samples = 2
        for step in ['pole', 'schild']:
            n_steps = 1 if step == 'pole' else 100
            tol = 1e-6 if step == 'pole' else 1e-1
            base_point = self.hypersphere.random_uniform(n_samples)
            tan_vec_a = self.hypersphere.projection_to_tangent_space(
                gs.random.rand(n_samples, 3), base_point)
            tan_vec_b = self.hypersphere.projection_to_tangent_space(
                gs.random.rand(n_samples, 3), base_point)
            expected = self.hypersphere.metric.parallel_transport(
                tan_vec_a, tan_vec_b, base_point)
            ladder = self.hypersphere.metric.ladder_parallel_transport(
                tan_vec_a, tan_vec_b, base_point, step=step, n_steps=n_steps)
            result = ladder['transported_tangent_vec']

            self.assertAllClose(result, expected, rtol=tol, atol=tol)

    @geomstats.tests.np_and_pytorch_only
    def test_parallel_transport_trajectory(self):
        n_samples = 2
        for step in ['pole', 'schild']:
            n_steps = 1 if step == 'pole' else 100
            rtol = 1e-6 if step == 'pole' else 1e-1
            base_point = self.hypersphere.random_uniform(n_samples)
            tan_vec_a = self.hypersphere.projection_to_tangent_space(
                gs.random.rand(n_samples, 3), base_point)
            tan_vec_b = self.hypersphere.projection_to_tangent_space(
                gs.random.rand(n_samples, 3), base_point)
            expected = self.hypersphere.metric.parallel_transport(
                tan_vec_a, tan_vec_b, base_point)
            ladder = self.hypersphere.metric.ladder_parallel_transport(
                tan_vec_a, tan_vec_b, base_point, return_geodesics=True,
                step=step, n_steps=n_steps)
            result = ladder['transported_tangent_vec']

            self.assertAllClose(result, expected, rtol=rtol)

    @geomstats.tests.np_only
    def test_exp(self):
        point = gs.array([[gs.pi / 2, 0], [gs.pi / 6, gs.pi / 4]])
        vector = gs.array([[0.25, 0.5], [0.30, 0.2]])
        point_ext = self.hypersphere.spherical_to_extrinsic(point)
        vector_ext = self.hypersphere.tangent_spherical_to_extrinsic(vector,
                                                                     point)
        self.connection.christoffels = self.hypersphere.metric.christoffels
        expected = self.hypersphere.metric.exp(vector_ext, point_ext)
        result_spherical = self.connection.exp(
            vector, point, n_steps=50, step='rk4')
        result = self.hypersphere.spherical_to_extrinsic(result_spherical)

        self.assertAllClose(result, expected, rtol=1e-6)

    @geomstats.tests.np_only
    def test_log(self):
        base_point = gs.array([[gs.pi / 3, gs.pi / 4], [gs.pi / 2, gs.pi / 4]])
        point = gs.array([[1.0, gs.pi / 2], [gs.pi / 6, gs.pi / 3]])
        self.connection.christoffels = self.hypersphere.metric.christoffels
        vector = self.connection.log(
            point=point, base_point=base_point, n_steps=75, step='rk')
        result = self.hypersphere.tangent_spherical_to_extrinsic(
            vector, base_point)
        p_ext = self.hypersphere.spherical_to_extrinsic(base_point)
        q_ext = self.hypersphere.spherical_to_extrinsic(point)
        expected = self.hypersphere.metric.log(base_point=p_ext, point=q_ext)

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    geomstats.tests.main()
