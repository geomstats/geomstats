"""Unit tests for the Hypersphere."""

import random
from contextlib import nullcontext as does_not_raise

import pytest

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from geomstats.learning.frechet_mean import FrechetMean
from tests.conftest import TestCase
from tests.data_generation import LevelSetTestData, RiemannianMetricTestData
from tests.parametrizers import LevelSetParametrizer, RiemannianMetricParametrizer

MEAN_ESTIMATION_TOL = 5e-3
KAPPA_ESTIMATION_TOL = 3e-2
ONLINE_KMEANS_TOL = 2e-2


class TestHypersphere(TestCase, metaclass=LevelSetParametrizer):
    space = Hypersphere

    class TestDataHypersphere(LevelSetTestData):
        def replace_values_data(self):
            smoke_data = [
                dict(
                    dim=4,
                    points=gs.ones((3, 5)),
                    new_points=gs.zeros((2, 5)),
                    indcs=[True, False, True],
                    expected=gs.stack([gs.zeros(5), gs.ones(5), gs.zeros(5)]),
                )
            ]
            return self.generate_tests(smoke_data)

        def angle_to_extrinsic_data(self):
            smoke_data = [
                dict(
                    dim=1, point=gs.pi / 4, expected=gs.array([1.0, 1.0]) / gs.sqrt(2.0)
                ),
                dict(
                    dim=1,
                    point=gs.array([1.0 / 3, 0.0]) * gs.pi,
                    expected=gs.array([[1.0 / 2, gs.sqrt(3.0) / 2], [1.0, 0.0]]),
                ),
            ]
            return self.generate_tests(smoke_data)

        def extrinsic_to_angle_data(self):
            smoke_data = [
                dict(
                    dim=1, point=gs.array([1.0, 1.0]) / gs.sqrt(2.0), expected=gs.pi / 4
                ),
                dict(
                    dim=1,
                    point=gs.array([[1.0 / 2, gs.sqrt(3.0) / 2], [1.0, 0.0]]),
                    expected=gs.array([1.0 / 3, 0.0]) * gs.pi,
                ),
            ]
            return self.generate_tests(smoke_data)

        def spherical_to_extrinsic_data(self):
            smoke_data = [
                dict(
                    dim=2,
                    point=gs.array([gs.pi / 2, 0]),
                    expected=gs.array([1.0, 0.0, 0.0]),
                ),
                dict(
                    dim=2,
                    point=gs.array([[gs.pi / 2, 0], [gs.pi / 6, gs.pi / 4]]),
                    expected=gs.array(
                        [
                            [1.0, 0.0, 0.0],
                            [
                                gs.sqrt(2.0) / 4.0,
                                gs.sqrt(2.0) / 4.0,
                                gs.sqrt(3.0) / 2.0,
                            ],
                        ]
                    ),
                ),
            ]
            return self.generate_tests(smoke_data)

        def extrinsic_to_spherical_data(self):
            smoke_data = [
                dict(
                    dim=2,
                    point=gs.array([1.0, 0.0, 0.0]),
                    expected=gs.array([gs.pi / 2, 0]),
                ),
                dict(
                    dim=2,
                    point=gs.array(
                        [
                            [1.0, 0.0, 0.0],
                            [
                                gs.sqrt(2.0) / 4.0,
                                gs.sqrt(2.0) / 4.0,
                                gs.sqrt(3.0) / 2.0,
                            ],
                        ]
                    ),
                    expected=gs.array([[gs.pi / 2, 0], [gs.pi / 6, gs.pi / 4]]),
                ),
            ]
            return self.generate_tests(smoke_data)

        def random_von_mises_fisher_belongs_data(self):
            dim_list = random.sample(range(2, 10), 5)
            n_samples_list = random.sample(range(1, 10), 5)
            random_data = [
                dict(dim=dim, n_samples=n_samples)
                for dim, n_samples in zip(dim_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

        def random_von_mises_fisher_mean_data(self):
            dim_list = random.sample(range(2, 10), 5)
            smoke_data = [
                dict(
                    dim=dim,
                    n_points=100000,
                    kappa=10,
                    expected=gs.array([1.0] + [0.0] * dim),
                    atol=KAPPA_ESTIMATION_TOL,
                )
                for dim in dim_list
            ]
            return self.generate_tests(smoke_data)

        def tangent_extrinsic_to_spherical_raises_data(self):
            smoke_data = []
            dim_list = [2, 3]
            for dim in dim_list:
                space = Hypersphere(dim)
                base_point = space.point()
                tangent_vec = space.to_tangent(space.random_point(), base_point)
                if dim == 2:
                    expected = does_not_raise()
                else:
                    expected = pytest.raises(NotImplementedError)
                smoke_data.append(
                    dict(
                        dim=dim,
                        tangent_vec=tangent_vec,
                        base_point=base_point,
                        base_point_spherical=None,
                        expected=expected,
                    )
                )
                smoke_data.append(
                    dict(
                        dim=dim,
                        tangent_vec=tangent_vec,
                        base_point=None,
                        base_point_spherical=None,
                        expected=does_not_raise(),
                    )
                )
            return self.generate_tests(smoke_data)

        def tangent_spherical_to_extrinsic_data(self):
            smoke_data = [
                dict(
                    dim=2,
                    tangent_vec_spherical=gs.array([[0.25, 0.5], [0.3, 0.2]]),
                    base_point_spherical=gs.array([[gs.pi / 2, 0], [gs.pi / 2, 0]]),
                    expected=gs.array([[0, 0.5, -0.25], [0, 0.2, -0.3]]),
                )
            ]
            return self.generate_tests(smoke_data)

        def tangent_extrinsic_to_spherical_data(self):
            smoke_data = [
                dict(
                    dim=2,
                    tangent_vec=gs.array([[0, 0.5, -0.25], [0, 0.2, -0.3]]),
                    base_point=None,
                    base_point_spherical=gs.array([[gs.pi / 2, 0], [gs.pi / 2, 0]]),
                    expected=gs.array([[0.25, 0.5], [0.3, 0.2]]),
                ),
                dict(
                    dim=2,
                    tangent_vec=[0, 0.5, -0.25],
                    base_point=[1.0, 0.0, 0.0],
                    base_point_spherical=None,
                    expected=[0.25, 0.5],
                ),
            ]
            return self.generate_tests(smoke_data)

        def riemannian_normal_frechet_mean_data(self):
            smoke_data = [dict(dim=3), dict(dim=4)]
            return self.generate_tests(smoke_data)

        def riemannian_normal_and_belongs_data(self):
            smoke_data = [dict(dim=3, n_points=1), dict(dim=4, n_points=10)]
            return self.generate_tests(smoke_data)

    def test_replace_values(self, dim, points, new_points, indcs, expected):
        space = self.space(dim)
        result = space._replace_values(
            gs.array(points), gs.array(new_points), gs.array(indcs)
        )
        self.assertAllClose(result, expected)

    def test_angle_to_extrinsic(self, dim, point, expected):
        space = self.space(dim)
        result = space.angle_to_extrinsic(point)
        self.assertAllClose(result, expected)

    def test_extrinsic_to_angle(self, dim, point, expected):
        space = self.space(dim)
        result = space.extrinsic_to_angle(point)
        self.assertAllClose(result, expected)

    def test_spherical_to_extrinic(self, dim, point, expected):
        space = self.space(dim)
        result = space.spherical_to_extrinsic(point)
        self.assertAllClose(result, expected)

    def test_extrinsic_to_spherical(self, dim, point, expected):
        space = self.space(dim)
        result = space.extrinsic_to_spherical(point)
        self.assertAllClose(result, expected)

    def test_random_von_mises_fisher_belongs(self, dim, n_samples):
        space = self.space(dim)
        result = space.belongs(space.random_von_mises_fisher(n_samples))
        self.assertAllClose(gs.all(result), gs.array(True))

    def test_random_von_mises_fisher_mean(self, dim, kappa, n_samples, expected, atol):
        space = self.space(dim)
        points = space.random_von_mises_fisher(kappa=kappa, n_samples=n_samples)
        sum_points = gs.sum(points, axis=0)
        result = sum_points / gs.linalg.norm(sum_points)
        self.assertAllClose(result, expected, atol=atol)

    def test_tangent_spherical_to_extrinsic(
        self, dim, tangent_vec_spherical, base_point_spherical, expected
    ):
        space = self.space(dim)
        result = space.tangent_spherical_to_extrinsic(
            tangent_vec_spherical, base_point_spherical
        )
        self.assertAllClose(result, expected)

    def test_tangent_extrinsic_to_spherical(
        self, dim, tangent_vec, base_point, base_point_spherical, expected
    ):
        space = self.space(dim)
        result = space.tangent_extrinsic_to_spherical(
            tangent_vec, base_point, base_point_spherical
        )
        self.assertAllClose(result, expected)

    def test_tangent_extrinsic_to_spherical_raises(
        self, dim, tangent_vec, base_point, base_point_spherical, expected
    ):
        space = self.space(dim)
        with expected:
            space.tangent_extrinsic_to_spherical(
                tangent_vec, base_point, base_point_spherical
            )

    def test_tangent_extrinsic_to_spherical_inverse(
        self, dim, tangent_spherical, base_point_spherical
    ):

        space = self.space(dim)
        tangent_extrinsic = space.tangent_spherical_to_extrinsic(
            tangent_spherical, base_point_spherical
        )
        result = space.tangent_extrinsic_to_spherical(
            tangent_extrinsic, base_point_spherical=base_point_spherical
        )
        self.assertAllClose(result, tangent_spherical)

    @geomstats.tests.np_autograd_and_torch_only
    def test_riemannian_normal_frechet_mean(self, dim):
        space = self.space(dim)
        mean = space.random_uniform()
        precision = gs.eye(space.dim) * 10
        sample = space.random_riemannian_normal(mean, precision, 10000)
        estimator = FrechetMean(space.metric, method="adaptive")
        estimator.fit(sample)
        estimate = estimator.estimate_
        self.assertAllClose(estimate, mean, atol=1e-2)

    @geomstats.tests.np_autograd_and_torch_only
    def test_riemannian_normal_and_belongs(self, dim, n_points):
        space = self.space(dim)
        mean = space.random_uniform()
        cov = gs.eye(dim)
        sample = space.random_riemannian_normal(mean, cov, n_points)
        result = space.belongs(sample)
        self.assertTrue(gs.all(result))


class TestHypersphereMetric(TestCase, metaclas=RiemannianMetricParametrizer):
    metric = connection = HypersphereMetric

    class TestDataHypersphereMetric(RiemannianMetricTestData):
        dim_list = random.sample(range(2, 7), 5)
        metric_args_list = [(n,) for n in dim_list]
        shape_list = [(dim + 1,) for dim in dim_list]
        space_list = [Hypersphere(n) for n in dim_list]
        n_points_list = random.sample(range(1, 7), 5)
        n_samples_list = random.sample(range(1, 7), 5)
        n_points_a_list = random.sample(range(1, 7), 5)
        n_points_b_list = [1]
        batch_size_list = random.sample(range(2, 7), 5)
        alpha_list = [1] * 5
        n_rungs_list = [1] * 5
        scheme_list = ["pole"] * 5

        def log_exp_composition_data(self):
            # edge case: two very close points, base_point_2 and point_2,
            # form an angle < epsilon
            base_point = gs.array([1.0, 2.0, 3.0, 4.0, 6.0])
            base_point = base_point / gs.linalg.norm(base_point)
            point = base_point + 1e-4 * gs.array([-1.0, -2.0, 1.0, 1.0, 0.1])
            point = point / gs.linalg.norm(point)
            smoke_data = [dict(space_args=(4,), point=point, base_point=base_point)]
            return self._log_exp_composition_data(
                self.metric_args_list, self.space_list, self.n_samples_list, smoke_data
            )

        def test_inner_product(self):
            smoke_data = [
                dict(
                    dim=4,
                    tangent_vec_a=[1.0, 0.0, 0.0, 0.0, 0.0],
                    tangent_vec_b=[0.0, 1.0, 0.0, 0.0, 0.0],
                    base_point=[0.0, 0.0, 0.0, 0.0, 1.0],
                    expected=0.0,
                )
            ]
            return self.generate_tests(smoke_data)

        def dist_data(self):
            # smoke data is currently testing points at orthogonal
            point_a = gs.array([10.0, -2.0, -0.5, 0.0, 0.0])
            point_a = point_a / gs.linalg.norm(point_a)
            point_b = gs.array([2.0, 10, 0.0, 0.0, 0.0])
            point_b = point_b / gs.linalg.norm(point_b)
            smoke_data = [
                dict(dim=4, point_a=point_a, point_b=point_b, expected=gs.pi / 2)
            ]
            return self.generate_tests(smoke_data)

        def diameter_data(self):
            point_a = gs.array([[0.0, 0.0, 1.0]])
            point_b = gs.array([[1.0, 0.0, 0.0]])
            point_c = gs.array([[0.0, 0.0, -1.0]])
            smoke_data = [
                dict(dim=2, points=[point_a, point_b, point_c], expected=gs.pi)
            ]
            return self.generate_tests(smoke_data)

        def christoffels_shape_data(self):
            point = gs.array([[gs.pi / 2, 0], [gs.pi / 6, gs.pi / 4]])
            smoke_data = [dict(dim=2, point=point, expected=[2, 2, 2, 2])]
            return self.generate_tests(smoke_data)

        def sectional_curvature_data(self):
            dim_list = random.sample(range(2, 10), 5)
            n_samples_list = random.sample(range(1, 10), 5)
            random_data = []
            for dim, n_samples in zip(dim_list, n_samples_list):
                sphere = Hypersphere(dim)
                base_point = sphere.random_uniform()
                tangent_vec_a = sphere.to_tangent(
                    gs.random.rand(n_samples, sphere.dim + 1), base_point
                )
                tangent_vec_b = sphere.to_tangent(
                    gs.random.rand(n_samples, sphere.dim + 1), base_point
                )
                expected = gs.ones(1)  # try shape here
                random_data.append(
                    dict(
                        dim=dim,
                        tangent_vec_a=tangent_vec_a,
                        tangent_vec_b=tangent_vec_b,
                    ),
                    expected=expected,
                )
            return self.generate_tests(random_data)

        def pairwise_data(self):
            smoke_data = [
                dict(
                    point_a=1.0
                    / gs.sqrt(129.0)
                    * gs.array([10.0, -2.0, -5.0, 0.0, 0.0]),
                    point_b=1.0
                    / gs.sqrt(435.0)
                    * gs.array([1.0, -20.0, -5.0, 0.0, 3.0]),
                    expected=gs.array([[0.0, 1.24864502], [1.24864502, 0.0]]),
                    rtol=1e-3,
                )
            ]
            return self.generate_tests(smoke_data)

    def test_inner_product(
        self, dim, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        metric = self.metric(dim)
        result = metric.inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, expected)

    def test_dist(self, dim, point_a, point_b, expected):
        metric = self.metric(dim)
        result = metric.dist(gs.array(point_a), gs.array(point_b))
        self.assertAllClose(result, gs.array(expected))

    def test_dist_pairwise(self, dim, point_a, point_b, expected):
        metric = self.metric(dim)
        result = metric.dist_pairwise(gs.array(point_a), gs.array(point_a))
        self.assertAllClose(result, gs.array(expected))

    def test_diameter(self, dim, points, expected):
        metric = self.metric(dim)
        result = metric.diameter(gs.array(points))
        self.assertAllClose(result, gs.array(expected))

    def test_christoffels_shape(self, dim, point, expected):
        metric = self.metric(dim)
        result = metric.christoffels(point)
        self.assertAllClose(gs.shape(result), expected)

    def test_sectional_curvature(
        self, dim, tangnet_vec_a, tangent_vec_b, base_point, expected
    ):
        metric = self.metric(dim)
        result = metric.sectional_curvature(tangnet_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(result, expected)


# class TestHypersphere(geomstats.tests.TestCase):
#     def setup_method(self):
#         gs.random.seed(1234)

#         self.dimension = 4
#         self.space = Hypersphere(dim=self.dimension)
#         self.metric = self.space.metric
#         self.n_samples = 10

#     def test_exp_and_log_and_projection_to_tangent_space_general_case(self):
#         """Test Log and Exp.

#         Test that the Riemannian exponential
#         and the Riemannian logarithm are inverse.

#         Expect their composition to give the identity function.

#         NB: points on the n-dimensional sphere are
#         (n+1)-D vectors of norm 1.
#         """
#         # Riemannian Exp then Riemannian Log
#         # General case
#         # NB: Riemannian log gives a regularized tangent vector,
#         # so we take the norm modulo 2 * pi.
#         base_point = gs.array([0.0, -3.0, 0.0, 3.0, 4.0])
#         base_point = base_point / gs.linalg.norm(base_point)

#         vector = gs.array([3.0, 2.0, 0.0, 0.0, -1.0])
#         vector = self.space.to_tangent(vector=vector, base_point=base_point)

#         exp = self.metric.exp(tangent_vec=vector, base_point=base_point)
#         result = self.metric.log(point=exp, base_point=base_point)

#         expected = vector
#         norm_expected = gs.linalg.norm(expected)
#         regularized_norm_expected = gs.mod(norm_expected, 2 * gs.pi)
#         expected = expected / norm_expected * regularized_norm_expected

#         # The Log can be the opposite vector on the tangent space,
#         # whose Exp gives the base_point
#         are_close = gs.allclose(result, expected)
#         norm_2pi = gs.isclose(gs.linalg.norm(result - expected), 2 * gs.pi)
#         self.assertTrue(are_close or norm_2pi)

#     def test_exp_and_log_and_projection_to_tangent_space_edge_case(self):
#         """Test Log and Exp.

#         Test that the Riemannian exponential
#         and the Riemannian logarithm are inverse.

#         Expect their composition to give the identity function.

#         NB: points on the n-dimensional sphere are
#         (n+1)-D vectors of norm 1.
#         """
#         # Riemannian Exp then Riemannian Log
#         # Edge case: tangent vector has norm < epsilon
#         base_point = gs.array([10.0, -2.0, -0.5, 34.0, 3.0])
#         base_point = base_point / gs.linalg.norm(base_point)
#         vector = 1e-4 * gs.array([0.06, -51.0, 6.0, 5.0, 3.0])
#         vector = self.space.to_tangent(vector=vector, base_point=base_point)

#         exp = self.metric.exp(tangent_vec=vector, base_point=base_point)
#         result = self.metric.log(point=exp, base_point=base_point)
#         self.assertAllClose(result, vector)

#     def test_dist_pairwise(self):

#         point_a = 1.0 / gs.sqrt(129.0) * gs.array([10.0, -2.0, -5.0, 0.0, 0.0])
#         point_b = 1.0 / gs.sqrt(435.0) * gs.array([1.0, -20.0, -5.0, 0.0, 3.0])

#         point = gs.array([point_a, point_b])
#         result = self.metric.dist_pairwise(point)

#         expected = gs.array([[0.0, 1.24864502], [1.24864502, 0.0]])

#         self.assertAllClose(result, expected, rtol=1e-3)

#     def test_exp_and_dist_and_projection_to_tangent_space(self):
#         base_point = gs.array([16.0, -2.0, -2.5, 84.0, 3.0])
#         base_point = base_point / gs.linalg.norm(base_point)
#         vector = gs.array([9.0, 0.0, -1.0, -2.0, 1.0])
#         tangent_vec = self.space.to_tangent(vector=vector, base_point=base_point)

#         exp = self.metric.exp(tangent_vec=tangent_vec, base_point=base_point)
#         result = self.metric.dist(base_point, exp)
#         expected = gs.linalg.norm(tangent_vec) % (2 * gs.pi)
#         self.assertAllClose(result, expected)

#     def test_exp_and_dist_and_projection_to_tangent_space_vec(self):
#         base_point = gs.array(
#             [[16.0, -2.0, -2.5, 84.0, 3.0], [16.0, -2.0, -2.5, 84.0, 3.0]]
#         )

#         base_single_point = gs.array([16.0, -2.0, -2.5, 84.0, 3.0])
#         scalar_norm = gs.linalg.norm(base_single_point)

#         base_point = base_point / scalar_norm
#         vector = gs.array([[9.0, 0.0, -1.0, -2.0, 1.0], [9.0, 0.0, -1.0, -2.0, 1]])

#         tangent_vec = self.space.to_tangent(vector=vector, base_point=base_point)

#         exp = self.metric.exp(tangent_vec=tangent_vec, base_point=base_point)

#         result = self.metric.dist(base_point, exp)
#         expected = gs.linalg.norm(tangent_vec, axis=-1) % (2 * gs.pi)

#         self.assertAllClose(result, expected)

#     def test_sample_von_mises_fisher_arbitrary_mean(self):
#         """
#         Check that the maximum likelihood estimates of the mean and
#         concentration parameter are close to the real values. A first
#         estimation of the concentration parameter is obtained by a
#         closed-form expression and improved through the Newton method.
#         """
#         for dim in [2, 9]:
#             n_points = 10000
#             sphere = Hypersphere(dim)

#             # check mean value for concentrated distribution for different mean
#             kappa = 1000.0
#             mean = sphere.random_uniform()
#             points = sphere.random_von_mises_fisher(
#                 mu=mean, kappa=kappa, n_samples=n_points
#             )
#             sum_points = gs.sum(points, axis=0)
#             result = sum_points / gs.linalg.norm(sum_points)
#             expected = mean
#             self.assertAllClose(result, expected, atol=MEAN_ESTIMATION_TOL)

#     def test_random_von_mises_kappa(self):
#         # check concentration parameter for dispersed distribution
#         kappa = 1.0
#         n_points = 100000
#         for dim in [2, 9]:
#             sphere = Hypersphere(dim)
#             points = sphere.random_von_mises_fisher(kappa=kappa, n_samples=n_points)
#             sum_points = gs.sum(points, axis=0)
#             mean_norm = gs.linalg.norm(sum_points) / n_points
#             kappa_estimate = (
#                 mean_norm * (dim + 1.0 - mean_norm**2) / (1.0 - mean_norm**2)
#             )
#             kappa_estimate = gs.cast(kappa_estimate, gs.float64)
#             p = dim + 1
#             n_steps = 100
#             for _ in range(n_steps):
#                 bessel_func_1 = scipy.special.iv(p / 2.0, kappa_estimate)
#                 bessel_func_2 = scipy.special.iv(p / 2.0 - 1.0, kappa_estimate)
#                 ratio = bessel_func_1 / bessel_func_2
#                 denominator = 1.0 - ratio**2 - (p - 1.0) * ratio / kappa_estimate
#                 mean_norm = gs.cast(mean_norm, gs.float64)
#                 kappa_estimate = kappa_estimate - (ratio - mean_norm) / denominator
#             result = kappa_estimate
#             expected = kappa
#             self.assertAllClose(result, expected, atol=KAPPA_ESTIMATION_TOL)
