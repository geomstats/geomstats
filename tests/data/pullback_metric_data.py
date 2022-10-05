import geomstats.backend as gs
from geomstats.geometry.pullback_metric import PullbackMetric
from tests.data_generation import TestData


class PullbackMetricTestData(TestData):

    Metric = PullbackMetric

    def sphere_immersion_test_data(self):
        smoke_data = [
            dict(
                spherical_coords=gs.array([0.0, 0.0]),
                expected=gs.array([0.0, 0.0, 1.0]),
            ),
            dict(
                spherical_coords=gs.array([gs.pi, 0.0]),
                expected=gs.array([0.0, 0.0, -1.0]),
            ),
            dict(
                spherical_coords=gs.array([gs.pi / 2.0, gs.pi]),
                expected=gs.array([-1.0, 0.0, 0.0]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def sphere_immersion_and_spherical_to_extrinsic_test_data(self):
        smoke_data = [dict(dim=2, point=gs.array([0.0, 0.0]))]
        return self.generate_tests(smoke_data)

    def tangent_sphere_immersion_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                tangent_vec=gs.array([1.0, 0.0]),
                point=gs.array([gs.pi / 2.0, gs.pi / 2.0]),
                expected=gs.array([0.0, 0.0, -1.0]),
            ),
            dict(
                dim=2,
                tangent_vec=gs.array([0.0, 1.0]),
                point=gs.array([gs.pi / 2.0, gs.pi / 2.0]),
                expected=gs.array([-1.0, 0.0, 0.0]),
            ),
            dict(
                dim=2,
                tangent_vec=gs.array([1.0, 0.0]),
                point=gs.array([gs.pi / 2.0, 0.0]),
                expected=gs.array([0.0, 0.0, -1.0]),
            ),
            dict(
                dim=2,
                tangent_vec=gs.array([0.0, 1.0]),
                point=gs.array([gs.pi / 2.0, 0.0]),
                expected=gs.array([0.0, 1.0, 0.0]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def tangent_circle_immersion_test_data(self):
        smoke_data = [
            dict(
                dim=1,
                tangent_vec=gs.array([1.0]),
                point=gs.array([0.0]),
                expected=gs.array([0.0, 1.0]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def jacobian_circle_immersion_test_data(self):
        smoke_data = [
            dict(dim=1, pole=gs.array([0.0])),
            dict(dim=1, pole=gs.array([0.2])),
            dict(dim=1, pole=gs.array([4.0])),
        ]
        return self.generate_tests(smoke_data)

    def jacobian_sphere_immersion_test_data(self):
        smoke_data = [
            dict(dim=2, pole=gs.array([0.0, 0.0])),
            dict(dim=2, pole=gs.array([0.22, 0.1])),
            dict(dim=2, pole=gs.array([0.1, 0.88])),
        ]
        return self.generate_tests(smoke_data)

    def parallel_transport_and_sphere_parallel_transport_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                tangent_vec_a=gs.array([0.0, 1.0]),
                tangent_vec_b=gs.array([0.0, 1.0]),
                base_point=gs.array([gs.pi / 2.0, 0.0]),
            )
        ]
        return self.generate_tests(smoke_data)

    def sphere_metric_matrix_test_data(self):
        smoke_data = [
            dict(dim=2, base_point=gs.array([0.0, 0.0])),
            dict(dim=2, base_point=gs.array([1.0, 1.0])),
            dict(dim=2, base_point=gs.array([0.3, 0.8])),
        ]
        return self.generate_tests(smoke_data)

    def circle_metric_matrix_test_data(self):
        smoke_data = [
            dict(dim=1, base_point=gs.array([0.0])),
            dict(dim=1, base_point=gs.array([1.0])),
            dict(dim=1, base_point=gs.array([4.0])),
        ]
        return self.generate_tests(smoke_data)

    def inner_product_and_sphere_inner_product_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                tangent_vec_a=gs.array([0.0, 1.0]),
                tangent_vec_b=gs.array([0.0, 1.0]),
                base_point=gs.array([gs.pi / 2.0, 0.0]),
            ),
            dict(
                dim=2,
                tangent_vec_a=gs.array([0.4, 1.0]),
                tangent_vec_b=gs.array([0.2, 0.6]),
                base_point=gs.array([gs.pi / 2.0, 0.1]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def inverse_sphere_metric_matrix_test_data(self):
        smoke_data = [
            dict(dim=2, base_point=gs.array([0.6, -1.0])),
            dict(dim=2, base_point=gs.array([0.8, -0.8])),
        ]
        return self.generate_tests(smoke_data)

    def inner_product_derivative_matrix_s2_test_data(self):
        smoke_data = [
            dict(dim=2, base_point=gs.array([0.6, -1.0])),
            dict(dim=2, base_point=gs.array([0.8, -0.8])),
        ]
        return self.generate_tests(smoke_data)

    def inverse_circle_metric_matrix_test_data(self):
        smoke_data = [
            dict(dim=1, base_point=gs.array([0.6])),
            dict(dim=1, base_point=gs.array([0.8])),
        ]
        return self.generate_tests(smoke_data)

    def christoffels_and_sphere_christoffels_test_data(self):
        smoke_data = [
            dict(dim=2, base_point=gs.array([0.1, 0.2])),
            dict(dim=2, base_point=gs.array([0.7, 0.233])),
        ]
        return self.generate_tests(smoke_data)

    def christoffels_circle_test_data(self):
        smoke_data = [
            dict(dim=1, base_point=gs.array([0.1])),
            dict(dim=1, base_point=gs.array([0.7])),
        ]
        return self.generate_tests(smoke_data)

    def exp_and_sphere_exp_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                tangent_vec=gs.array([0.0, 1.0]),
                base_point=gs.array([gs.pi / 2.0, 0.0]),
            ),
            dict(
                dim=2,
                tangent_vec=gs.array([0.4, 1.0]),
                base_point=gs.array([gs.pi / 2.0, 0.1]),
            ),
        ]
        return self.generate_tests(smoke_data)
