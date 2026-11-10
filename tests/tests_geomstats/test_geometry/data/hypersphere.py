import random

import pytest
from polpo.testing.data import CaseData
from polpo.testing.geometric import GeometricCaseData

import geomstats.backend as gs

from .riemannian_metric import RiemannianMetricTestData


class Hypersphere2MetricTestData(CaseData):
    def hamiltonian_test_data(self):
        data = [
            dict(
                state=(gs.array([0.0, 0.0, 1.0]), gs.array([1.0, 2.0, 1.0])),
                expected=gs.array(3.0),
            )
        ]
        return data


class Hypersphere4MetricTestData(CaseData):
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                tangent_vec_b=gs.array([0.0, 1.0, 0.0, 0.0, 0.0]),
                base_point=gs.array([0.0, 0.0, 0.0, 0.0, 1.0]),
                expected=gs.array(0.0),
            ),
        ]
        return data

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
        return data

    def exp_and_dist_and_projection_to_tangent_space_test_data(self):
        unnorm_base_point = gs.array([16.0, -2.0, -2.5, 84.0, 3.0])
        base_point = unnorm_base_point / gs.linalg.norm(unnorm_base_point)
        data = [
            dict(
                vector=gs.array([9.0, 0.0, -1.0, -2.0, 1.0]),
                base_point=base_point,
            )
        ]
        return data


class HypersphereMetricTestData(RiemannianMetricTestData):
    def sectional_curvature_is_one_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )


class CircleAnglesMetricTestData(RiemannianMetricTestData):
    @pytest.mark.smoke
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
        return data

    @pytest.mark.smoke
    def log_test_data(self):
        data = [
            dict(
                point=gs.array([-3 * gs.pi / 4]),
                base_point=gs.array([3 * gs.pi / 4]),
                expected=gs.array([gs.pi / 2]),
            ),
        ]
        return data

    @pytest.mark.smoke
    def dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([3 * gs.pi / 4]),
                point_b=gs.array([-3 * gs.pi / 4]),
                expected=gs.array(gs.pi / 2),
            ),
        ]
        return data


class SphereSphericalCoordinatesMetricTestData(GeometricCaseData):
    @pytest.mark.smoke
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([0.3, 0.4]),
                tangent_vec_b=gs.array([0.1, -0.5]),
                base_point=gs.array([gs.pi / 3.0, gs.pi / 5.0]),
                expected=gs.array(-0.12),
            )
        ]
        return data

    def christoffels_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="base_point",
            on_metric=True,
        )

    @pytest.mark.smoke
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
        return data

    @pytest.mark.smoke
    def riemann_tensor_spherical_coords_test_data(self):
        data = [dict(base_point=gs.array([gs.pi / 2, gs.pi / 6]))]
        return data

    @pytest.mark.smoke
    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([gs.pi / 2.0, 0.0]),
                base_point=gs.array([gs.pi / 10.0, gs.pi / 9.0]),
                expected=gs.array([gs.pi / 10.0 + gs.pi / 2.0, gs.pi / 9.0]),
            )
        ]
        return data


class DistributionTestData(CaseData):
    @pytest.mark.random
    def sample_belongs_test_data(self):
        return [dict(n_samples=n_samples) for n_samples in [1, random.randint(2, 5)]]


class VonMisesFisherTestData(DistributionTestData):
    @pytest.mark.random
    def sample_mean_test_data(self):
        return [dict(n_samples=1000, atol=1e-2)]


class VonMisesFisherKappaTestData(CaseData):
    @pytest.mark.validation
    def sample_kappa_test_data(self):
        return [dict(n_samples=5000, atol=1e-1)]


class HypersphereRiemannianNormalFrechetMeanTestData(CaseData):
    @pytest.mark.random
    def sample_frechet_mean_test_data(self):
        return [dict(n_samples=5000, atol=1e-1)]
