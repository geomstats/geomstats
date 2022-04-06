"""Unit tests for the Hypersphere."""

import random
from contextlib import nullcontext as does_not_raise

import pytest
import scipy.special

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from geomstats.learning.frechet_mean import FrechetMean
from tests.conftest import Parametrizer
from tests.data_generation import _LevelSetTestData, _RiemannianMetricTestData
from tests.geometry_test_cases import LevelSetTestCase, RiemannianMetricTestCase

MEAN_ESTIMATION_TOL = 1e-1
KAPPA_ESTIMATION_TOL = 1e-1
ONLINE_KMEANS_TOL = 1e-1


class TestHypersphere(LevelSetTestCase, metaclass=Parametrizer):
    space = Hypersphere

    class HypersphereTestData(_LevelSetTestData):

        dim_list = random.sample(range(1, 4), 2)
        space_args_list = [(dim,) for dim in dim_list]
        n_points_list = random.sample(range(1, 5), 2)
        shape_list = [(dim + 1,) for dim in dim_list]
        n_vecs_list = random.sample(range(1, 5), 2)

        def replace_values_test_data(self):
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

        def angle_to_extrinsic_test_data(self):
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

        def extrinsic_to_angle_test_data(self):
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

        def spherical_to_extrinsic_test_data(self):
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

        def extrinsic_to_spherical_test_data(self):
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

        def random_von_mises_fisher_belongs_test_data(self):
            dim_list = random.sample(range(2, 8), 5)
            n_samples_list = random.sample(range(1, 10), 5)
            random_data = [
                dict(dim=dim, n_samples=n_samples)
                for dim, n_samples in zip(dim_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

        def random_von_mises_fisher_mean_test_data(self):
            dim_list = random.sample(range(2, 8), 5)
            smoke_data = [
                dict(
                    dim=dim,
                    kappa=10,
                    n_points=100000,
                    expected=gs.array([1.0] + [0.0] * dim),
                    atol=KAPPA_ESTIMATION_TOL,
                )
                for dim in dim_list
            ]
            return self.generate_tests(smoke_data)

        def tangent_extrinsic_to_spherical_raises_test_data(self):
            smoke_data = []
            dim_list = [2, 3]
            for dim in dim_list:
                space = Hypersphere(dim)
                base_point = space.random_point()
                tangent_vec = space.to_tangent(space.random_point(), base_point)
                if dim == 2:
                    expected = does_not_raise()
                    smoke_data.append(
                        dict(
                            dim=2,
                            tangent_vec=tangent_vec,
                            base_point=None,
                            base_point_spherical=None,
                            expected=pytest.raises(ValueError),
                        )
                    )
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

            return self.generate_tests(smoke_data)

        def tangent_spherical_to_extrinsic_test_data(self):
            smoke_data = [
                dict(
                    dim=2,
                    tangent_vec_spherical=gs.array([[0.25, 0.5], [0.3, 0.2]]),
                    base_point_spherical=gs.array([[gs.pi / 2, 0], [gs.pi / 2, 0]]),
                    expected=gs.array([[0, 0.5, -0.25], [0, 0.2, -0.3]]),
                )
            ]
            return self.generate_tests(smoke_data)

        def tangent_extrinsic_to_spherical_test_data(self):
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
                    tangent_vec=gs.array([0, 0.5, -0.25]),
                    base_point=gs.array([1.0, 0.0, 0.0]),
                    base_point_spherical=None,
                    expected=gs.array([0.25, 0.5]),
                ),
            ]
            return self.generate_tests(smoke_data)

        def riemannian_normal_frechet_mean_test_data(self):
            smoke_data = [dict(dim=3), dict(dim=4)]
            return self.generate_tests(smoke_data)

        def riemannian_normal_and_belongs_test_data(self):
            smoke_data = [dict(dim=3, n_points=1), dict(dim=4, n_points=10)]
            return self.generate_tests(smoke_data)

        def sample_von_mises_fisher_mean_test_data(self):
            dim_list = random.sample(range(2, 10), 5)
            smoke_data = [
                dict(
                    dim=dim,
                    mean=Hypersphere(dim).random_point(),
                    kappa=1000.0,
                    n_points=10000,
                )
                for dim in dim_list
            ]
            return self.generate_tests(smoke_data)

        def sample_random_von_mises_fisher_kappa_test_data(self):
            dim_list = random.sample(range(2, 8), 5)
            smoke_data = [dict(dim=dim, kappa=1.0, n_points=50000) for dim in dim_list]
            return self.generate_tests(smoke_data)

        def random_point_belongs_test_data(self):
            belongs_atol = gs.atol * 10000
            smoke_space_args_list = [(2,), (3,), (4,)]
            smoke_n_points_list = [1, 2, 1]
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
                belongs_atol,
            )

        def to_tangent_is_tangent_test_data(self):

            is_tangent_atol = gs.atol * 1000
            return self._to_tangent_is_tangent_test_data(
                Hypersphere,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
                is_tangent_atol,
            )

        def projection_belongs_test_data(self):
            return self._projection_belongs_test_data(
                self.space_args_list, self.shape_list, self.n_points_list
            )

        def extrinsic_then_intrinsic_test_data(self):
            space_args_list = [(1,), (2,)]
            return self._extrinsic_then_intrinsic_test_data(
                Hypersphere, space_args_list, self.n_points_list, atol=gs.atol * 100
            )

        def intrinsic_then_extrinsic_test_data(self):
            space_args_list = [(1,), (2,)]
            return self._intrinsic_then_extrinsic_test_data(
                Hypersphere, space_args_list, self.n_points_list, atol=gs.atol * 100
            )

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                Hypersphere, self.space_args_list, self.n_vecs_list
            )

    testing_data = HypersphereTestData()

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

    def test_spherical_to_extrinsic(self, dim, point, expected):
        space = self.space(dim)
        result = space.spherical_to_extrinsic(point)
        self.assertAllClose(result, expected)

    def test_extrinsic_to_spherical(self, dim, point, expected):
        space = self.space(dim)
        result = space.extrinsic_to_spherical(point)
        self.assertAllClose(result, expected)

    def test_random_von_mises_fisher_belongs(self, dim, n_samples):
        space = self.space(dim)
        result = space.belongs(space.random_von_mises_fisher(n_samples=n_samples))
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

    @geomstats.tests.np_autograd_and_torch_only
    def test_riemannian_normal_frechet_mean(self, dim):
        space = self.space(dim)
        mean = space.random_uniform()
        precision = gs.eye(space.dim) * 10
        sample = space.random_riemannian_normal(mean, precision, 30000)
        estimator = FrechetMean(space.metric, method="adaptive")
        estimator.fit(sample)
        estimate = estimator.estimate_
        self.assertAllClose(estimate, mean, atol=1e-1)

    @geomstats.tests.np_autograd_and_torch_only
    def test_riemannian_normal_and_belongs(self, dim, n_points):
        space = self.space(dim)
        mean = space.random_uniform()
        cov = gs.eye(dim)
        sample = space.random_riemannian_normal(mean, cov, n_points)
        result = space.belongs(sample)
        self.assertTrue(gs.all(result))

    def test_sample_von_mises_fisher_mean(self, dim, mean, kappa, n_points):
        """
        Check that the maximum likelihood estimates of the mean and
        concentration parameter are close to the real values. A first
        estimation of the concentration parameter is obtained by a
        closed-form expression and improved through the Newton method.
        """
        space = self.space(dim)
        points = space.random_von_mises_fisher(mu=mean, kappa=kappa, n_samples=n_points)
        sum_points = gs.sum(points, axis=0)
        result = sum_points / gs.linalg.norm(sum_points)
        expected = mean
        self.assertAllClose(result, expected, atol=MEAN_ESTIMATION_TOL)

    def test_sample_random_von_mises_fisher_kappa(self, dim, kappa, n_points):
        # check concentration parameter for dispersed distribution
        sphere = Hypersphere(dim)
        points = sphere.random_von_mises_fisher(kappa=kappa, n_samples=n_points)
        sum_points = gs.sum(points, axis=0)
        mean_norm = gs.linalg.norm(sum_points) / n_points
        kappa_estimate = (
            mean_norm * (dim + 1.0 - mean_norm**2) / (1.0 - mean_norm**2)
        )
        kappa_estimate = gs.cast(kappa_estimate, gs.float64)
        p = dim + 1
        n_steps = 100
        for _ in range(n_steps):
            bessel_func_1 = scipy.special.iv(p / 2.0, kappa_estimate)
            bessel_func_2 = scipy.special.iv(p / 2.0 - 1.0, kappa_estimate)
            ratio = bessel_func_1 / bessel_func_2
            denominator = 1.0 - ratio**2 - (p - 1.0) * ratio / kappa_estimate
            mean_norm = gs.cast(mean_norm, gs.float64)
            kappa_estimate = kappa_estimate - (ratio - mean_norm) / denominator
        result = kappa_estimate
        expected = kappa
        self.assertAllClose(result, expected, atol=KAPPA_ESTIMATION_TOL)


class TestHypersphereMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = HypersphereMetric
    skip_test_exp_geodesic_ivp = True
    skip_test_dist_point_to_itself_is_zero = True

    class HypersphereMetricTestData(_RiemannianMetricTestData):
        dim_list = random.sample(range(2, 5), 2)
        metric_args_list = [(n,) for n in dim_list]
        shape_list = [(dim + 1,) for dim in dim_list]
        space_list = [Hypersphere(n) for n in dim_list]
        n_points_list = random.sample(range(1, 5), 2)
        n_tangent_vecs_list = random.sample(range(1, 5), 2)
        n_points_a_list = random.sample(range(1, 5), 2)
        n_points_b_list = [1]
        alpha_list = [1] * 2
        n_rungs_list = [1] * 2
        scheme_list = ["pole"] * 2

        def inner_product_test_data(self):
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

        def dist_test_data(self):
            # smoke data is currently testing points at orthogonal
            point_a = gs.array([10.0, -2.0, -0.5, 0.0, 0.0])
            point_a = point_a / gs.linalg.norm(point_a)
            point_b = gs.array([2.0, 10, 0.0, 0.0, 0.0])
            point_b = point_b / gs.linalg.norm(point_b)
            smoke_data = [
                dict(dim=4, point_a=point_a, point_b=point_b, expected=gs.pi / 2)
            ]
            return self.generate_tests(smoke_data)

        def diameter_test_data(self):
            point_a = gs.array([[0.0, 0.0, 1.0]])
            point_b = gs.array([[1.0, 0.0, 0.0]])
            point_c = gs.array([[0.0, 0.0, -1.0]])
            smoke_data = [
                dict(
                    dim=2, points=gs.vstack((point_a, point_b, point_c)), expected=gs.pi
                )
            ]
            return self.generate_tests(smoke_data)

        def christoffels_shape_test_data(self):
            point = gs.array([[gs.pi / 2, 0], [gs.pi / 6, gs.pi / 4]])
            smoke_data = [dict(dim=2, point=point, expected=[2, 2, 2, 2])]
            return self.generate_tests(smoke_data)

        def sectional_curvature_test_data(self):
            dim_list = [4]
            n_samples_list = random.sample(range(1, 4), 2)
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
                expected = gs.ones(n_samples)  # try shape here
                random_data.append(
                    dict(
                        dim=dim,
                        tangent_vec_a=tangent_vec_a,
                        tangent_vec_b=tangent_vec_b,
                        base_point=base_point,
                        expected=expected,
                    ),
                )
            return self.generate_tests(random_data)

        def dist_pairwise_test_data(self):
            smoke_data = [
                dict(
                    dim=4,
                    point=[
                        1.0 / gs.sqrt(129.0) * gs.array([10.0, -2.0, -5.0, 0.0, 0.0]),
                        1.0 / gs.sqrt(435.0) * gs.array([1.0, -20.0, -5.0, 0.0, 3.0]),
                    ],
                    expected=gs.array([[0.0, 1.24864502], [1.24864502, 0.0]]),
                    rtol=1e-3,
                )
            ]
            return self.generate_tests(smoke_data)

        def exp_shape_test_data(self):
            return self._exp_shape_test_data(
                self.metric_args_list, self.space_list, self.shape_list
            )

        def log_shape_test_data(self):
            return self._log_shape_test_data(self.metric_args_list, self.space_list)

        def squared_dist_is_symmetric_test_data(self):
            return self._squared_dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
                atol=gs.atol * 1000,
            )

        def exp_belongs_test_data(self):
            return self._exp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_is_tangent_test_data(self):
            return self._log_is_tangent_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                is_tangent_atol=gs.atol * 1000,
            )

        def geodesic_ivp_belongs_test_data(self):
            return self._geodesic_ivp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def geodesic_bvp_belongs_test_data(self):
            return self._geodesic_bvp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def exp_after_log_test_data(self):
            # edge case: two very close points, base_point_2 and point_2,
            # form an angle < epsilon
            base_point = gs.array([1.0, 2.0, 3.0, 4.0, 6.0])
            base_point = base_point / gs.linalg.norm(base_point)
            point = base_point + 1e-4 * gs.array([-1.0, -2.0, 1.0, 1.0, 0.1])
            point = point / gs.linalg.norm(point)
            smoke_data = [
                dict(
                    space_args=(4,),
                    point=point,
                    base_point=base_point,
                    rtol=gs.rtol,
                    atol=gs.atol,
                )
            ]
            return self._exp_after_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                smoke_data,
                atol=1e-3,
            )

        def log_after_exp_test_data(self):
            base_point = gs.array([1.0, 0.0, 0.0, 0.0])
            tangent_vec = gs.array([0.0, 0.0, gs.pi / 6, 0.0])

            smoke_data = [
                dict(
                    space_args=(4,),
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    rtol=gs.rtol,
                    atol=gs.atol,
                )
            ]
            return self._log_after_exp_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                smoke_data,
                amplitude=gs.pi / 2.0,
                rtol=gs.rtol,
                atol=gs.atol,
            )

        def exp_ladder_parallel_transport_test_data(self):
            return self._exp_ladder_parallel_transport_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_rungs_list,
                self.alpha_list,
                self.scheme_list,
            )

        def exp_geodesic_ivp_test_data(self):
            return self._exp_geodesic_ivp_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_points_list,
                rtol=1e-3,
                atol=1e-3,
            )

        def parallel_transport_ivp_is_isometry_test_data(self):
            return self._parallel_transport_ivp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def parallel_transport_bvp_is_isometry_test_data(self):
            return self._parallel_transport_bvp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def dist_is_symmetric_test_data(self):
            return self._dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_positive_test_data(self):
            return self._dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def squared_dist_is_positive_test_data(self):
            return self._squared_dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_norm_of_log_test_data(self):
            return self._dist_is_norm_of_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_point_to_itself_is_zero_test_data(self):
            return self._dist_point_to_itself_is_zero_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                atol=gs.atol * 10000,
            )

        def inner_product_is_symmetric_test_data(self):
            return self._inner_product_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
            )

        def exp_and_dist_and_projection_to_tangent_space_test_data(self):
            unnorm_base_point = gs.array([16.0, -2.0, -2.5, 84.0, 3.0])
            base_point = unnorm_base_point / gs.linalg.norm(unnorm_base_point)
            smoke_data = [
                dict(
                    dim=4,
                    vector=gs.array([9.0, 0.0, -1.0, -2.0, 1.0]),
                    base_point=base_point,
                )
            ]
            return self.generate_tests(smoke_data)

    testing_data = HypersphereMetricTestData()

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

    def test_dist_pairwise(self, dim, point, expected, rtol):
        metric = self.metric(dim)
        result = metric.dist_pairwise(gs.array(point))
        self.assertAllClose(result, gs.array(expected), rtol=rtol)

    def test_diameter(self, dim, points, expected):
        metric = self.metric(dim)
        result = metric.diameter(gs.array(points))
        self.assertAllClose(result, gs.array(expected))

    def test_christoffels_shape(self, dim, point, expected):
        metric = self.metric(dim)
        result = metric.christoffels(point)
        self.assertAllClose(gs.shape(result), expected)

    def test_sectional_curvature(
        self, dim, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        metric = self.metric(dim)
        result = metric.sectional_curvature(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(result, expected, atol=1e-2)

    def test_exp_and_dist_and_projection_to_tangent_space(
        self, dim, vector, base_point
    ):
        metric = self.metric(dim)
        tangent_vec = Hypersphere(dim).to_tangent(vector=vector, base_point=base_point)
        exp = metric.exp(tangent_vec=tangent_vec, base_point=base_point)
        result = metric.dist(base_point, exp)
        expected = gs.linalg.norm(tangent_vec) % (2 * gs.pi)
        self.assertAllClose(result, expected)
