import geomstats.backend as gs
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.data_generation import TestData


class ConnectionTestData(TestData):
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
                base_point=gs.array([[gs.pi / 3, gs.pi / 4], [gs.pi / 2, gs.pi / 4]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def geodesic_with_exp_connection_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point=gs.array([1.0, gs.pi / 2]),
                tangent_vec=gs.array([gs.pi / 3, gs.pi / 4]),
                n_times=10,
                n_steps=10,
                expected=(10, 2),
            ),
            dict(
                dim=2,
                point=gs.array([1.0, gs.pi / 2]),
                tangent_vec=gs.array([[gs.pi / 3, gs.pi / 4], [gs.pi / 2, -gs.pi / 4]]),
                n_times=10,
                n_steps=100,
                expected=(2, 10, 2),
            ),
            dict(
                dim=2,
                point=gs.array([[1.0, gs.pi / 2], [gs.pi / 6, gs.pi / 3]]),
                tangent_vec=gs.array([[gs.pi / 3, gs.pi / 4], [gs.pi / 2, gs.pi / 4]]),
                n_times=10,
                n_steps=100,
                expected=(2, 10, 2),
            ),
        ]
        return self.generate_tests(smoke_data)

    def geodesic_with_log_connection_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point=gs.array([1.0, gs.pi / 2]),
                end_point=gs.array([gs.pi / 3, gs.pi / 4]),
                n_times=10,
                n_steps=10,
                expected=(10, 2),
            ),
            dict(
                dim=2,
                point=gs.array([[1.0, gs.pi / 2], [gs.pi / 6, gs.pi / 3]]),
                end_point=gs.array([[gs.pi / 3, gs.pi / 4], [gs.pi / 2, gs.pi / 4]]),
                n_times=10,
                n_steps=100,
                expected=(2, 10, 2),
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
