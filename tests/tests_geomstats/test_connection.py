"""Unit tests for the affine connections."""

import warnings

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.connection import Connection
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class TestConnection(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=UserWarning)
        gs.random.seed(0)
        self.dim = 4
        self.euc_metric = EuclideanMetric(dim=self.dim)

        self.connection = Connection(dim=2)
        self.hypersphere = Hypersphere(dim=2)

    def test_metric_matrix(self):
        base_point = gs.array([0.0, 1.0, 0.0, 0.0])

        result = self.euc_metric.metric_matrix(base_point)
        expected = gs.eye(self.dim)

        self.assertAllClose(result, expected)

    def test_parallel_transport(self):
        n_samples = 2
        base_point = self.hypersphere.random_uniform(n_samples)
        tan_vec_a = self.hypersphere.to_tangent(
            gs.random.rand(n_samples, 3), base_point
        )
        tan_vec_b = self.hypersphere.to_tangent(
            gs.random.rand(n_samples, 3), base_point
        )
        expected = self.hypersphere.metric.parallel_transport(
            tan_vec_a, tan_vec_b, base_point
        )
        expected_point = self.hypersphere.metric.exp(tan_vec_b, base_point)
        base_point = gs.cast(base_point, gs.float64)
        base_point, tan_vec_a, tan_vec_b = gs.convert_to_wider_dtype(
            [base_point, tan_vec_a, tan_vec_b]
        )
        for step, alpha in zip(["pole", "schild"], [1, 2]):
            min_n = 1 if step == "pole" else 50
            tol = 1e-5 if step == "pole" else 1e-2
            for n_rungs in [min_n, 11]:
                ladder = self.hypersphere.metric.ladder_parallel_transport(
                    tan_vec_a,
                    tan_vec_b,
                    base_point,
                    scheme=step,
                    n_rungs=n_rungs,
                    alpha=alpha,
                )
                result = ladder["transported_tangent_vec"]
                result_point = ladder["end_point"]
                self.assertAllClose(result, expected, rtol=tol, atol=tol)
                self.assertAllClose(result_point, expected_point)

    def test_parallel_transport_trajectory(self):
        n_samples = 2
        for step in ["pole", "schild"]:
            n_steps = 1 if step == "pole" else 50
            tol = 1e-6 if step == "pole" else 1e-2
            base_point = self.hypersphere.random_uniform(n_samples)
            tan_vec_a = self.hypersphere.to_tangent(
                gs.random.rand(n_samples, 3), base_point
            )
            tan_vec_b = self.hypersphere.to_tangent(
                gs.random.rand(n_samples, 3), base_point
            )
            expected = self.hypersphere.metric.parallel_transport(
                tan_vec_a, tan_vec_b, base_point
            )
            expected_point = self.hypersphere.metric.exp(tan_vec_b, base_point)
            ladder = self.hypersphere.metric.ladder_parallel_transport(
                tan_vec_a,
                tan_vec_b,
                base_point,
                return_geodesics=True,
                scheme=step,
                n_rungs=n_steps,
            )
            result = ladder["transported_tangent_vec"]
            result_point = ladder["end_point"]

            self.assertAllClose(result, expected, rtol=tol, atol=tol)
            self.assertAllClose(result_point, expected_point)

    def test_ladder_alpha(self):
        n_samples = 2
        base_point = self.hypersphere.random_uniform(n_samples)
        tan_vec_a = self.hypersphere.to_tangent(
            gs.random.rand(n_samples, 3), base_point
        )
        tan_vec_b = self.hypersphere.to_tangent(
            gs.random.rand(n_samples, 3), base_point
        )
        self.assertRaises(
            ValueError,
            lambda: self.hypersphere.metric.ladder_parallel_transport(
                tan_vec_a,
                tan_vec_b,
                base_point,
                return_geodesics=False,
                scheme="pole",
                n_rungs=1,
                alpha=0.5,
            ),
        )

    def test_exp_connection_metric(self):
        point = gs.array([gs.pi / 2, 0])
        vector = gs.array([0.25, 0.5])
        point_ext = self.hypersphere.spherical_to_extrinsic(point)
        vector_ext = self.hypersphere.tangent_spherical_to_extrinsic(vector, point)
        self.connection.christoffels = self.hypersphere.metric.christoffels
        expected = self.hypersphere.metric.exp(vector_ext, point_ext)
        result_spherical = self.connection.exp(vector, point, n_steps=50, step="rk4")
        result = self.hypersphere.spherical_to_extrinsic(result_spherical)

        self.assertAllClose(result, expected)

    def test_exp_connection_metric_vectorization(self):
        point = gs.array([[gs.pi / 2, 0], [gs.pi / 6, gs.pi / 4]])
        vector = gs.array([[0.25, 0.5], [0.30, 0.2]])
        point_ext = self.hypersphere.spherical_to_extrinsic(point)
        vector_ext = self.hypersphere.tangent_spherical_to_extrinsic(vector, point)
        self.connection.christoffels = self.hypersphere.metric.christoffels
        expected = self.hypersphere.metric.exp(vector_ext, point_ext)
        result_spherical = self.connection.exp(vector, point, n_steps=50, step="rk4")
        result = self.hypersphere.spherical_to_extrinsic(result_spherical)

        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_log_connection_metric(self):
        base_point = gs.array([gs.pi / 3, gs.pi / 4])
        point = gs.array([1.0, gs.pi / 2])
        self.connection.christoffels = self.hypersphere.metric.christoffels
        vector = self.connection.log(
            point=point, base_point=base_point, n_steps=75, step="rk4", tol=1e-10
        )
        result = self.hypersphere.tangent_spherical_to_extrinsic(vector, base_point)
        p_ext = self.hypersphere.spherical_to_extrinsic(base_point)
        q_ext = self.hypersphere.spherical_to_extrinsic(point)
        expected = self.hypersphere.metric.log(base_point=p_ext, point=q_ext)

        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_log_connection_metric_vectorization(self):
        base_point = gs.array([[gs.pi / 3, gs.pi / 4], [gs.pi / 2, gs.pi / 4]])
        point = gs.array([[1.0, gs.pi / 2], [gs.pi / 6, gs.pi / 3]])
        self.connection.christoffels = self.hypersphere.metric.christoffels
        vector = self.connection.log(
            point=point, base_point=base_point, n_steps=75, step="rk4", tol=1e-10
        )
        result = self.hypersphere.tangent_spherical_to_extrinsic(vector, base_point)
        p_ext = self.hypersphere.spherical_to_extrinsic(base_point)
        q_ext = self.hypersphere.spherical_to_extrinsic(point)
        expected = self.hypersphere.metric.log(base_point=p_ext, point=q_ext)

        self.assertAllClose(result, expected, atol=1e-6)

    def test_geodesic_and_coincides_exp_hypersphere(self):
        n_geodesic_points = 10
        initial_point = self.hypersphere.random_uniform(2)
        vector = gs.array([[2.0, 0.0, -1.0]] * 2)
        initial_tangent_vec = self.hypersphere.to_tangent(
            vector=vector, base_point=initial_point
        )
        geodesic = self.hypersphere.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )
        t = gs.linspace(start=0.0, stop=1.0, num=n_geodesic_points)
        points = geodesic(t)
        result = points[:, -1]
        expected = self.hypersphere.metric.exp(vector, initial_point)
        self.assertAllClose(expected, result)

        initial_point = initial_point[0]
        initial_tangent_vec = initial_tangent_vec[0]
        geodesic = self.hypersphere.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )
        points = geodesic(t)
        result = points[-1]
        expected = self.hypersphere.metric.exp(initial_tangent_vec, initial_point)
        self.assertAllClose(expected, result)

    def test_geodesic_and_coincides_exp_son(self):
        n_geodesic_points = 10
        space = SpecialOrthogonal(n=4)
        initial_point = space.random_uniform(2)
        vector = gs.random.rand(2, 4, 4)
        initial_tangent_vec = space.to_tangent(vector=vector, base_point=initial_point)
        geodesic = space.bi_invariant_metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )
        t = gs.linspace(start=0.0, stop=1.0, num=n_geodesic_points)
        points = geodesic(t)
        result = points[:, -1]
        expected = space.bi_invariant_metric.exp(initial_tangent_vec, initial_point)
        self.assertAllClose(result, expected)

        initial_point = initial_point[0]
        initial_tangent_vec = initial_tangent_vec[0]
        geodesic = space.bi_invariant_metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )
        points = geodesic(t)
        result = points[-1]
        expected = space.bi_invariant_metric.exp(initial_tangent_vec, initial_point)
        self.assertAllClose(expected, result)

    def test_geodesic_invalid_initial_conditions(self):
        space = SpecialOrthogonal(n=4)
        initial_point = space.random_uniform(2)
        vector = gs.random.rand(2, 4, 4)
        initial_tangent_vec = space.to_tangent(vector=vector, base_point=initial_point)
        end_point = space.random_uniform(2)
        self.assertRaises(
            RuntimeError,
            lambda: space.bi_invariant_metric.geodesic(
                initial_point=initial_point,
                initial_tangent_vec=initial_tangent_vec,
                end_point=end_point,
            ),
        )

    def test_geodesic_vectorization(self):
        space = Hypersphere(2)
        metric = space.metric
        initial_point = space.random_uniform(2)
        vector = gs.random.rand(2, 3)
        initial_tangent_vec = space.to_tangent(vector=vector, base_point=initial_point)
        end_point = space.random_uniform(2)
        time = gs.linspace(0, 1, 10)

        geo = metric.geodesic(initial_point, initial_tangent_vec)
        path = geo(time)
        result = path.shape
        expected = (2, 10, 3)
        self.assertAllClose(result, expected)

        geo = metric.geodesic(initial_point, end_point=end_point)
        path = geo(time)
        result = path.shape
        expected = (2, 10, 3)
        self.assertAllClose(result, expected)

        geo = metric.geodesic(initial_point, end_point=end_point[0])
        path = geo(time)
        result = path.shape
        expected = (2, 10, 3)
        self.assertAllClose(result, expected)

        initial_tangent_vec = space.to_tangent(
            vector=vector, base_point=initial_point[0]
        )
        geo = metric.geodesic(initial_point[0], initial_tangent_vec)
        path = geo(time)
        result = path.shape
        expected = (2, 10, 3)
        self.assertAllClose(result, expected)
