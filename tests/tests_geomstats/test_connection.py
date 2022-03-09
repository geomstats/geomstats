"""Unit tests for the affine connections."""


import pytest

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.connection import Connection
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.conftest import TestCase
from tests.data_generation import TestData
from tests.parametrizers import Parametrizer


class TestConnection(TestCase, metaclass=Parametrizer):
    class TestDataConnection(TestData):
        def metric_matrix_test_data(self):
            smoke_data = [
                dict(
                    metric=EuclideanMetric(dim=4),
                    point=gs.array([0.0, 1.0, 0.0, 0.0]),
                    expected=gs.eye(4),
                )
            ]
            return self.generate_tests(smoke_data)

        def parallel_transport_test_data(self):
            smoke_data = [dict(dim=2, n_sample=2)]
            return self.generate_tests(smoke_data)

        def parallel_transport_trajectory_test_data(self):
            smoke_data = [dict(dim=2, n_sample=2)]
            return self.generate_tests(smoke_data)

        def exp_connection_metric_test_data(self):
            smoke_data = [
                dict(
                    dim=2,
                    tangent_vec=gs.array([[0.25, 0.5], [0.30, 0.2]]),
                    point=gs.array([[gs.pi / 2, 0], [gs.pi / 6, gs.pi / 4]]),
                ),
                dict(
                    dim=2,
                    tangent_vec=gs.array([0.25, 0.5]),
                    point=gs.array([gs.pi / 2, 0]),
                ),
            ]
            return self.generate_tests(smoke_data)

        def log_connection_metric_test_data(self):
            smoke_data = [
                dict(
                    dim=2,
                    point=gs.array([1.0, gs.pi / 2]),
                    base_point=gs.array([gs.pi / 3, gs.pi / 4]),
                ),
                dict(
                    dim=2,
                    point=gs.array([[1.0, gs.pi / 2], [gs.pi / 6, gs.pi / 3]]),
                    base_point=gs.array(
                        [[gs.pi / 3, gs.pi / 4], [gs.pi / 2, gs.pi / 4]]
                    ),
                ),
            ]
            return self.generate_tests(smoke_data)

        def geodesic_and_coincides_exp_test_data(self):
            smoke_data = [
                dict(
                    space=Hypersphere(2),
                    n_geodesic_points=10,
                    vec=gs.array([[2.0, 0.0, -1.0]] * 2),
                ),
                dict(
                    space=SpecialOrthogonal(n=4),
                    n_geodesic_points=10,
                    vec=gs.random.rand(2, 4, 4),
                ),
            ]
            return self.generate_tests(smoke_data)

        def geodesic_invalid_initial_conditions_test_data(self):
            smoke_data = [dict(space=SpecialOrthogonal(n=4))]
            return self.generate_tests(smoke_data)

        def geodesic_test_data(self):
            smoke_data = [dict(space=Hypersphere(2))]
            return self.generate_tests(smoke_data)

        def ladder_alpha_test_data(self):
            smoke_data = [dict(dim=2, n_samples=2)]
            return self.generate_tests(smoke_data)

    testing_data = TestDataConnection()

    def test_metric_matrix(self, metric, point, expected):
        self.assertAllClose(metric.metric_matrix(point), expected)

    def test_parallel_transport(self, dim, n_samples):
        sphere = Hypersphere(dim)
        base_point = sphere.random_uniform(n_samples)
        tan_vec_a = sphere.to_tangent(gs.random.rand(n_samples, 3), base_point)
        tan_vec_b = sphere.to_tangent(gs.random.rand(n_samples, 3), base_point)
        expected = sphere.metric.parallel_transport(tan_vec_a, base_point, tan_vec_b)
        expected_point = sphere.metric.exp(tan_vec_b, base_point)
        base_point = gs.cast(base_point, gs.float64)
        base_point, tan_vec_a, tan_vec_b = gs.convert_to_wider_dtype(
            [base_point, tan_vec_a, tan_vec_b]
        )
        for step, alpha in zip(["pole", "schild"], [1, 2]):
            min_n = 1 if step == "pole" else 50
            tol = 1e-5 if step == "pole" else 1e-2
            for n_rungs in [min_n, 11]:
                ladder = sphere.metric.ladder_parallel_transport(
                    tan_vec_a,
                    base_point,
                    tan_vec_b,
                    n_rungs=n_rungs,
                    scheme=step,
                    alpha=alpha,
                )
                result = ladder["transported_tangent_vec"]
                result_point = ladder["end_point"]
                self.assertAllClose(result, expected, rtol=tol, atol=tol)
                self.assertAllClose(result_point, expected_point)

    def test_parallel_transport_trajectory(self, dim, n_samples):
        sphere = Hypersphere(dim)
        for step in ["pole", "schild"]:
            n_steps = 1 if step == "pole" else 50
            tol = 1e-6 if step == "pole" else 1e-2
            base_point = sphere.random_uniform(n_samples)
            tan_vec_a = sphere.to_tangent(gs.random.rand(n_samples, 3), base_point)
            tan_vec_b = sphere.to_tangent(gs.random.rand(n_samples, 3), base_point)
            expected = sphere.metric.parallel_transport(
                tan_vec_a, base_point, tan_vec_b
            )
            expected_point = sphere.metric.exp(tan_vec_b, base_point)
            ladder = sphere.metric.ladder_parallel_transport(
                tan_vec_a,
                base_point,
                tan_vec_b,
                n_rungs=n_steps,
                scheme=step,
                return_geodesics=True,
            )
            result = ladder["transported_tangent_vec"]
            result_point = ladder["end_point"]

            self.assertAllClose(result, expected, rtol=tol, atol=tol)
            self.assertAllClose(result_point, expected_point)

    def test_ladder_alpha(self, dim, n_samples):
        sphere = Hypersphere(dim)
        base_point = sphere.random_uniform(n_samples)
        tan_vec_a = sphere.to_tangent(gs.random.rand(n_samples, 3), base_point)
        tan_vec_b = sphere.to_tangent(gs.random.rand(n_samples, 3), base_point)

        with pytest.raises(ValueError):
            sphere.metric.ladder_parallel_transport(
                tan_vec_a,
                base_point,
                tan_vec_b,
                n_rungs=1,
                scheme="pole",
                alpha=0.5,
                return_geodesics=False,
            )

    def test_exp_connection_metric(self, dim, tangent_vec, base_point):
        sphere = Hypersphere(dim)
        connection = Connection(dim)
        point_ext = sphere.spherical_to_extrinsic(base_point)
        vector_ext = sphere.tangent_spherical_to_extrinsic(tangent_vec, base_point)
        connection.christoffels = sphere.metric.christoffels
        expected = sphere.metric.exp(vector_ext, point_ext)
        result_spherical = connection.exp(
            tangent_vec, base_point, n_steps=50, step="rk4"
        )
        result = sphere.spherical_to_extrinsic(result_spherical)

        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_log_connection_metric(self, dim, point, base_point):
        sphere = Hypersphere(dim)
        connection = Connection(dim)
        connection.christoffels = sphere.metric.christoffels
        vector = connection.log(
            point=point, base_point=base_point, n_steps=75, step="rk4", tol=1e-10
        )
        result = sphere.tangent_spherical_to_extrinsic(vector, base_point)
        p_ext = sphere.spherical_to_extrinsic(base_point)
        q_ext = sphere.spherical_to_extrinsic(point)
        expected = sphere.metric.log(base_point=p_ext, point=q_ext)

        self.assertAllClose(result, expected, atol=1e-6)

    def test_geodesic_and_coincides_exp(self, space, n_geodesic_points, vector):
        initial_point = space.random_uniform(2)
        initial_tangent_vec = space.to_tangent(vector=vector, base_point=initial_point)
        geodesic = space.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )
        t = gs.linspace(start=0.0, stop=1.0, num=n_geodesic_points)
        points = geodesic(t)
        result = points[:, -1]
        expected = space.metric.exp(initial_tangent_vec, initial_point)
        self.assertAllClose(result, expected)

        initial_point = initial_point[0]
        initial_tangent_vec = initial_tangent_vec[0]
        geodesic = space.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )
        points = geodesic(t)
        result = points[-1]
        expected = space.metric.exp(initial_tangent_vec, initial_point)
        self.assertAllClose(expected, result)

    def test_geodesic_invalid_initial_conditions(self, space):
        initial_point = space.random_uniform(2)
        vector = gs.random.rand(2, 4, 4)
        initial_tangent_vec = space.to_tangent(vector=vector, base_point=initial_point)
        end_point = space.random_uniform(2)
        with pytest.raises(RuntimeError):
            space.bi_invariant_metric.geodesic(
                initial_point=initial_point,
                initial_tangent_vec=initial_tangent_vec,
                end_point=end_point,
            )

    def test_geodesic(self, space):
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
