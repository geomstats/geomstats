import pytest

import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import DiffeomorphicManifoldTestData
from .riemannian_metric import (
    RiemannianMetricCmpWithTransformTestData,
    RiemannianMetricTestData,
)


class Hypersphere2MetricTestData(TestData):
    def hamiltonian_test_data(self):
        data = [
            dict(
                state=(gs.array([0.0, 0.0, 1.0]), gs.array([1.0, 2.0, 1.0])),
                expected=gs.array(3.0),
            )
        ]
        return self.generate_tests(data)


class Hypersphere4MetricTestData(TestData):
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                tangent_vec_b=gs.array([0.0, 1.0, 0.0, 0.0, 0.0]),
                base_point=gs.array([0.0, 0.0, 0.0, 0.0, 1.0]),
                expected=gs.array(0.0),
            ),
        ]
        return self.generate_tests(data)

    def dist_test_data(self):
        point_a = gs.array([10.0, -2.0, -0.5, 0.0, 0.0])
        point_a = point_a / gs.linalg.norm(point_a)
        point_b = gs.array([2.0, 10, 0.0, 0.0, 0.0])
        point_b = point_b / gs.linalg.norm(point_b)
        data = [
            dict(
                point_a=point_a,
                point_b=point_b,
                expected=gs.array(gs.pi / 2),
            ),
            dict(
                point_a=gs.expand_dims(
                    1.0 / gs.sqrt(129.0) * gs.array([10.0, -2.0, -5.0, 0.0, 0.0]),
                    axis=0,
                ),
                point_b=gs.expand_dims(
                    1.0 / gs.sqrt(435.0) * gs.array([1.0, -20.0, -5.0, 0.0, 3.0]),
                    axis=0,
                ),
                expected=gs.array([1.24864502]),
            ),
        ]
        return self.generate_tests(data)

    def exp_and_dist_and_projection_to_tangent_space_test_data(self):
        unnorm_base_point = gs.array([16.0, -2.0, -2.5, 84.0, 3.0])
        base_point = unnorm_base_point / gs.linalg.norm(unnorm_base_point)
        data = [
            dict(
                vector=gs.array([9.0, 0.0, -1.0, -2.0, 1.0]),
                base_point=base_point,
            )
        ]
        return self.generate_tests(data)


class HypersphereMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    trials = 3

    tolerances = {"dist_point_to_itself_is_zero": {"atol": 1e-6}}

    def sectional_curvature_is_one_test_data(self):
        return self.generate_random_data()


class CircleAnglesMetricTestData(RiemannianMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False

    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([gs.pi / 2]),
                base_point=gs.array([gs.pi / 4]),
                expected=gs.array([3 * gs.pi / 4]),
            ),
            dict(
                tangent_vec=gs.array([gs.pi / 2]),
                base_point=gs.array([3 * gs.pi / 4]),
                expected=gs.array([-3 * gs.pi / 4]),
            ),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([-3 * gs.pi / 4]),
                base_point=gs.array([3 * gs.pi / 4]),
                expected=gs.array([gs.pi / 2]),
            ),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))

    def dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([3 * gs.pi / 4]),
                point_b=gs.array([-3 * gs.pi / 4]),
                expected=gs.array(gs.pi / 2),
            ),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))


class CircleAnglesMetricCmpTestData(RiemannianMetricCmpWithTransformTestData):
    skips = (
        "curvature_random",
        "sectional_curvature_random",
        "scalar_curvature_random",
        "inner_product_derivative_matrix_random",
        "inner_coproduct_random",
    )


class SphereSphericalCoordinatesTestData(DiffeomorphicManifoldTestData):
    skips = ("not_belongs",)


class SphereSphericalCoordinatesMetricCmpTestData(
    RiemannianMetricCmpWithTransformTestData
):
    skips = (
        "scalar_curvature_random",
        "inner_product_derivative_matrix_random",
        "inner_coproduct_random",
    )


class SphereSphericalCoordinatesMetricTestData(TestData):
    fail_for_autodiff_exceptions = False

    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([0.3, 0.4]),
                tangent_vec_b=gs.array([0.1, -0.5]),
                base_point=gs.array([gs.pi / 3.0, gs.pi / 5.0]),
                expected=gs.array(-0.12),
            )
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))

    def christoffels_vec_test_data(self):
        return self.generate_vec_data()

    def ricci_tensor_test_data(self):
        theta = gs.pi / 3
        base_point = gs.array([theta, gs.pi / 7])
        expected = gs.array([[1.0, 0.0], [0.0, gs.sin(theta) ** 2]])

        base_points = gs.array(
            [[gs.pi / 3, gs.pi / 7], [gs.pi / 4, gs.pi / 8], [gs.pi / 5, gs.pi / 9]]
        )
        expected_ = gs.array(
            [
                [[1.0, 0.0], [0.0, gs.sin(gs.pi / 3) ** 2]],
                [[1.0, 0.0], [0.0, gs.sin(gs.pi / 4) ** 2]],
                [[1.0, 0.0], [0.0, gs.sin(gs.pi / 5) ** 2]],
            ]
        )
        data = [
            dict(base_point=base_point, expected=expected),
            dict(base_point=base_points, expected=expected_),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))

    def riemann_tensor_spherical_coords_test_data(self):
        data = [dict(base_point=gs.array([gs.pi / 2, gs.pi / 6]))]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))

    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([gs.pi / 2.0, 0.0]),
                base_point=gs.array([gs.pi / 10.0, gs.pi / 9.0]),
                expected=gs.array([gs.pi / 10.0 + gs.pi / 2.0, gs.pi / 9.0]),
            )
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))


class DistributionTestData(TestData):
    def sample_belongs_test_data(self):
        data = [dict(n_samples=n_samples) for n_samples in self.N_RANDOM_POINTS]
        return self.generate_tests(data)


class VonMisesFisherTestData(DistributionTestData):
    tolerances = {"sample_mean": {"atol": 1e-2}}

    def sample_mean_test_data(self):
        return self.generate_tests([dict(n_samples=1000)])


class VonMisesFisherKappaTestData(TestData):
    tolerances = {"sample_kappa": {"atol": 1e-1}}

    def sample_kappa_test_data(self):
        return self.generate_tests([dict(n_samples=5000)])


class HypersphereRiemannianNormalFrechetMeanTestData(TestData):
    tolerances = {"sample_frechet_mean": {"atol": 1e-1}}

    def sample_frechet_mean_test_data(self):
        return self.generate_tests([dict(n_samples=5000)])
