"""Unit tests for the Hypersphere."""

import random

import pytest
import scipy.special

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.learning.frechet_mean import FrechetMean
from tests.conftest import TestCase
from tests.data_generation import LevelSetTestData, RiemannianMetricTestData, TestData
from tests.parametrizers import (
    LevelSetParametrizer,
    ManifoldParametrizer,
    RiemannianMetricParametrizer,
)

MEAN_ESTIMATION_TOL = 5e-3
KAPPA_ESTIMATION_TOL = 3e-2
ONLINE_KMEANS_TOL = 2e-2


class TestHypersphere(TestCase, metaclass=LevelSetParametrizer):
    space = Hypersphere

    class TestDataHypersphere(TestData):
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

        def spherical_to_extrinsic():
            smoke_data = [dict(dim=2, point=gs.array([gs.pi / 2, 0]),expected=gs.array([1.0, 0.0, 0.0])), dict(dim=2,point=gs.array([[gs.pi / 2, 0], [gs.pi / 6, gs.pi / 4]]), expected= gs.array(
            [
                [1.0, 0.0, 0.0],
                [gs.sqrt(2.0) / 4.0, gs.sqrt(2.0) / 4.0, gs.sqrt(3.0) / 2.0],
            ]
        ))]

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

    def test_spherical_to_extrinsic(self):
        """
        Check vectorization of conversion from spherical
        to extrinsic coordinates on the 2-sphere.
        """
        dim = 2
        sphere = Hypersphere(dim)

        points_spherical = gs.array([gs.pi / 2, 0])
        result = sphere.spherical_to_extrinsic(points_spherical)
        expected = gs.array([1.0, 0.0, 0.0])
        self.assertAllClose(result, expected)

    def test_extrinsic_to_spherical(self):
        """
        Check vectorization of conversion from spherical
        to extrinsic coordinates on the 2-sphere.
        """
        dim = 2
        sphere = Hypersphere(dim)

        points_extrinsic = gs.array([1.0, 0.0, 0.0])
        result = sphere.extrinsic_to_spherical(points_extrinsic)
        expected = gs.array([gs.pi / 2, 0])
        self.assertAllClose(result, expected)

    def test_spherical_to_extrinsic_vectorization(self):
        dim = 2
        sphere = Hypersphere(dim)
        points_spherical = gs.array([[gs.pi / 2, 0], [gs.pi / 6, gs.pi / 4]])
        result = sphere.spherical_to_extrinsic(points_spherical)
        expected =
        self.assertAllClose(result, expected)

    def test_extrinsic_to_spherical_vectorization(self):
        dim = 2
        sphere = Hypersphere(dim)
        expected = gs.array([[gs.pi / 2, 0], [gs.pi / 6, gs.pi / 4]])
        point_extrinsic = gs.array(
            [
                [1.0, 0.0, 0.0],
                [gs.sqrt(2.0) / 4.0, gs.sqrt(2.0) / 4.0, gs.sqrt(3.0) / 2.0],
            ]
        )
        result = sphere.extrinsic_to_spherical(point_extrinsic)
        self.assertAllClose(result, expected)

    def test_spherical_to_extrinsic_and_inverse(self):
        dim = 2
        n_samples = 5
        sphere = Hypersphere(dim)
        points = gs.random.rand(n_samples, 2) * gs.pi * gs.array([1.0, 2.0])[None, :]
        extrinsic = sphere.spherical_to_extrinsic(points)
        result = sphere.extrinsic_to_spherical(extrinsic)
        self.assertAllClose(result, points)

        points_extrinsic = sphere.random_uniform(n_samples)
        spherical = sphere.extrinsic_to_spherical(points_extrinsic)
        result = sphere.spherical_to_extrinsic(spherical)
        self.assertAllClose(result, points_extrinsic)


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
            # following smoke_data covers edge case: two very close points, base_point_2 and point_2,
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


class TestHypersphere(geomstats.tests.TestCase):
    def setup_method(self):
        gs.random.seed(1234)

        self.dimension = 4
        self.space = Hypersphere(dim=self.dimension)
        self.metric = self.space.metric
        self.n_samples = 10

    def test_exp_and_log_and_projection_to_tangent_space_general_case(self):
        """Test Log and Exp.

        Test that the Riemannian exponential
        and the Riemannian logarithm are inverse.

        Expect their composition to give the identity function.

        NB: points on the n-dimensional sphere are
        (n+1)-D vectors of norm 1.
        """
        # Riemannian Exp then Riemannian Log
        # General case
        # NB: Riemannian log gives a regularized tangent vector,
        # so we take the norm modulo 2 * pi.
        base_point = gs.array([0.0, -3.0, 0.0, 3.0, 4.0])
        base_point = base_point / gs.linalg.norm(base_point)

        vector = gs.array([3.0, 2.0, 0.0, 0.0, -1.0])
        vector = self.space.to_tangent(vector=vector, base_point=base_point)

        exp = self.metric.exp(tangent_vec=vector, base_point=base_point)
        result = self.metric.log(point=exp, base_point=base_point)

        expected = vector
        norm_expected = gs.linalg.norm(expected)
        regularized_norm_expected = gs.mod(norm_expected, 2 * gs.pi)
        expected = expected / norm_expected * regularized_norm_expected

        # The Log can be the opposite vector on the tangent space,
        # whose Exp gives the base_point
        are_close = gs.allclose(result, expected)
        norm_2pi = gs.isclose(gs.linalg.norm(result - expected), 2 * gs.pi)
        self.assertTrue(are_close or norm_2pi)

    def test_exp_and_log_and_projection_to_tangent_space_edge_case(self):
        """Test Log and Exp.

        Test that the Riemannian exponential
        and the Riemannian logarithm are inverse.

        Expect their composition to give the identity function.

        NB: points on the n-dimensional sphere are
        (n+1)-D vectors of norm 1.
        """
        # Riemannian Exp then Riemannian Log
        # Edge case: tangent vector has norm < epsilon
        base_point = gs.array([10.0, -2.0, -0.5, 34.0, 3.0])
        base_point = base_point / gs.linalg.norm(base_point)
        vector = 1e-4 * gs.array([0.06, -51.0, 6.0, 5.0, 3.0])
        vector = self.space.to_tangent(vector=vector, base_point=base_point)

        exp = self.metric.exp(tangent_vec=vector, base_point=base_point)
        result = self.metric.log(point=exp, base_point=base_point)
        self.assertAllClose(result, vector)

    def test_dist_pairwise(self):

        point_a = 1.0 / gs.sqrt(129.0) * gs.array([10.0, -2.0, -5.0, 0.0, 0.0])
        point_b = 1.0 / gs.sqrt(435.0) * gs.array([1.0, -20.0, -5.0, 0.0, 3.0])

        point = gs.array([point_a, point_b])
        result = self.metric.dist_pairwise(point)

        expected = gs.array([[0.0, 1.24864502], [1.24864502, 0.0]])

        self.assertAllClose(result, expected, rtol=1e-3)

    def test_dist_pairwise_parallel(self):
        n_samples = 15
        points = self.space.random_uniform(n_samples)
        result = self.metric.dist_pairwise(points, n_jobs=2, prefer="threads")
        is_sym = Matrices.is_symmetric(result)
        belongs = Matrices(n_samples, n_samples).belongs(result)
        self.assertTrue(is_sym)
        self.assertTrue(belongs)

    def test_exp_and_dist_and_projection_to_tangent_space(self):
        base_point = gs.array([16.0, -2.0, -2.5, 84.0, 3.0])
        base_point = base_point / gs.linalg.norm(base_point)
        vector = gs.array([9.0, 0.0, -1.0, -2.0, 1.0])
        tangent_vec = self.space.to_tangent(vector=vector, base_point=base_point)

        exp = self.metric.exp(tangent_vec=tangent_vec, base_point=base_point)
        result = self.metric.dist(base_point, exp)
        expected = gs.linalg.norm(tangent_vec) % (2 * gs.pi)
        self.assertAllClose(result, expected)

    def test_exp_and_dist_and_projection_to_tangent_space_vec(self):
        base_point = gs.array(
            [[16.0, -2.0, -2.5, 84.0, 3.0], [16.0, -2.0, -2.5, 84.0, 3.0]]
        )

        base_single_point = gs.array([16.0, -2.0, -2.5, 84.0, 3.0])
        scalar_norm = gs.linalg.norm(base_single_point)

        base_point = base_point / scalar_norm
        vector = gs.array([[9.0, 0.0, -1.0, -2.0, 1.0], [9.0, 0.0, -1.0, -2.0, 1]])

        tangent_vec = self.space.to_tangent(vector=vector, base_point=base_point)

        exp = self.metric.exp(tangent_vec=tangent_vec, base_point=base_point)

        result = self.metric.dist(base_point, exp)
        expected = gs.linalg.norm(tangent_vec, axis=-1) % (2 * gs.pi)

        self.assertAllClose(result, expected)

    def test_closest_neighbor_index(self):
        """Check that the closest neighbor is one of neighbors."""
        n_samples = 10
        points = self.space.random_uniform(n_samples=n_samples)
        point = points[0, :]
        neighbors = points[1:, :]
        index = self.metric.closest_neighbor_index(point, neighbors)
        closest_neighbor = points[index, :]

        test = gs.sum(gs.all(points == closest_neighbor, axis=1))
        result = test > 0
        self.assertTrue(result)

    def test_sample_von_mises_fisher_arbitrary_mean(self):
        """
        Check that the maximum likelihood estimates of the mean and
        concentration parameter are close to the real values. A first
        estimation of the concentration parameter is obtained by a
        closed-form expression and improved through the Newton method.
        """
        for dim in [2, 9]:
            n_points = 10000
            sphere = Hypersphere(dim)

            # check mean value for concentrated distribution for different mean
            kappa = 1000.0
            mean = sphere.random_uniform()
            points = sphere.random_von_mises_fisher(
                mu=mean, kappa=kappa, n_samples=n_points
            )
            sum_points = gs.sum(points, axis=0)
            result = sum_points / gs.linalg.norm(sum_points)
            expected = mean
            self.assertAllClose(result, expected, atol=MEAN_ESTIMATION_TOL)

    def test_random_von_mises_kappa(self):
        # check concentration parameter for dispersed distribution
        kappa = 1.0
        n_points = 100000
        for dim in [2, 9]:
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

    def test_random_von_mises_general_dim_mean(self):
        for dim in [2, 9]:
            sphere = Hypersphere(dim)
            n_points = 100000

            # check mean value for concentrated distribution
            kappa = 10
            points = sphere.random_von_mises_fisher(kappa=kappa, n_samples=n_points)
            sum_points = gs.sum(points, axis=0)
            expected = gs.array([1.0] + [0.0] * dim)
            result = sum_points / gs.linalg.norm(sum_points)
            self.assertAllClose(result, expected, atol=KAPPA_ESTIMATION_TOL)

    def test_random_von_mises_one_sample_belongs(self):
        for dim in [2, 9]:
            sphere = Hypersphere(dim)
            point = sphere.random_von_mises_fisher()
            self.assertAllClose(point.shape, (dim + 1,))
            result = sphere.belongs(point)
            self.assertTrue(result)

    def test_tangent_spherical_to_extrinsic(self):
        """
        Check vectorization of conversion from spherical
        to extrinsic coordinates for tangent vectors to the
        2-sphere.
        """
        dim = 2
        sphere = Hypersphere(dim)
        base_points_spherical = gs.array([[gs.pi / 2, 0], [gs.pi / 2, 0]])
        tangent_vecs_spherical = gs.array([[0.25, 0.5], [0.3, 0.2]])
        result = sphere.tangent_spherical_to_extrinsic(
            tangent_vecs_spherical, base_points_spherical
        )
        expected = gs.array([[0, 0.5, -0.25], [0, 0.2, -0.3]])
        self.assertAllClose(result, expected)

        result = sphere.tangent_spherical_to_extrinsic(
            tangent_vecs_spherical[0], base_points_spherical[0]
        )
        self.assertAllClose(result, expected[0])

    def test_tangent_extrinsic_to_spherical(self):
        """
        Check vectorization of conversion from spherical
        to extrinsic coordinates for tangent vectors to the
        2-sphere.
        """
        dim = 2
        sphere = Hypersphere(dim)
        base_points_spherical = gs.array([[gs.pi / 2, 0], [gs.pi / 2, 0]])
        expected = gs.array([[0.25, 0.5], [0.3, 0.2]])
        tangent_vecs = gs.array([[0, 0.5, -0.25], [0, 0.2, -0.3]])
        result = sphere.tangent_extrinsic_to_spherical(
            tangent_vecs, base_point_spherical=base_points_spherical
        )
        self.assertAllClose(result, expected)

        result = sphere.tangent_extrinsic_to_spherical(
            tangent_vecs[0], base_point=gs.array([1.0, 0.0, 0.0])
        )
        self.assertAllClose(result, expected[0])

    def test_tangent_spherical_and_extrinsic_inverse(self):
        dim = 2
        n_samples = 5
        sphere = Hypersphere(dim)
        points = gs.random.rand(n_samples, 2) * gs.pi * gs.array([1.0, 2.0])[None, :]
        tangent_spherical = gs.random.rand(n_samples, 2)
        tangent_extrinsic = sphere.tangent_spherical_to_extrinsic(
            tangent_spherical, points
        )
        result = sphere.tangent_extrinsic_to_spherical(
            tangent_extrinsic, base_point_spherical=points
        )
        self.assertAllClose(result, tangent_spherical)

        points_extrinsic = sphere.random_uniform(n_samples)
        vector = gs.random.rand(n_samples, dim + 1)
        tangent_extrinsic = sphere.to_tangent(vector, points_extrinsic)
        tangent_spherical = sphere.tangent_extrinsic_to_spherical(
            tangent_extrinsic, base_point=points_extrinsic
        )
        spherical = sphere.extrinsic_to_spherical(points_extrinsic)
        result = sphere.tangent_spherical_to_extrinsic(tangent_spherical, spherical)
        self.assertAllClose(result, tangent_extrinsic)

    def test_christoffels_vectorization(self):
        """
        Check vectorization of Christoffel symbols in
        spherical coordinates on the 2-sphere.
        """
        dim = 2
        sphere = Hypersphere(dim)
        points_spherical = gs.array([[gs.pi / 2, 0], [gs.pi / 6, gs.pi / 4]])
        christoffel = sphere.metric.christoffels(points_spherical)
        result = christoffel.shape
        expected = gs.array([2, dim, dim, dim])
        self.assertAllClose(result, expected)

    def test_sectional_curvature(self):
        n_samples = 4
        sphere = self.space
        base_point = sphere.random_uniform(n_samples)
        tan_vec_a = sphere.to_tangent(
            gs.random.rand(n_samples, sphere.dim + 1), base_point
        )
        tan_vec_b = sphere.to_tangent(
            gs.random.rand(n_samples, sphere.dim + 1), base_point
        )
        result = sphere.metric.sectional_curvature(tan_vec_a, tan_vec_b, base_point)
        expected = gs.ones(result.shape)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_riemannian_normal_and_belongs(self):
        mean = self.space.random_uniform()
        cov = gs.eye(self.space.dim)
        sample = self.space.random_riemannian_normal(mean, cov, 10)
        result = self.space.belongs(sample)
        self.assertTrue(gs.all(result))

    @geomstats.tests.np_autograd_and_torch_only
    def test_riemannian_normal_mean(self):
        space = self.space
        mean = space.random_uniform()
        precision = gs.eye(space.dim) * 10
        sample = space.random_riemannian_normal(mean, precision, 10000)
        estimator = FrechetMean(space.metric, method="adaptive")
        estimator.fit(sample)
        estimate = estimator.estimate_
        self.assertAllClose(estimate, mean, atol=1e-2)

    def test_raises(self):
        space = self.space
        point = space.random_uniform()
        with pytest.raises(NotImplementedError):
            space.extrinsic_to_spherical(point)

        with pytest.raises(NotImplementedError):
            space.tangent_extrinsic_to_spherical(point, point)

        sphere = Hypersphere(2)

        with pytest.raises(ValueError):
            sphere.tangent_extrinsic_to_spherical(point)

    def test_extrinsic_to_angle_inverse(self):
        space = Hypersphere(1)
        point = space.random_uniform()
        angle = space.extrinsic_to_angle(point)
        result = space.angle_to_extrinsic(angle)
        self.assertAllClose(result, point)

        space = Hypersphere(1, default_coords_type="intrinsic")
        angle = space.random_uniform()
        extrinsic = space.angle_to_extrinsic(angle)
        result = space.extrinsic_to_angle(extrinsic)
        self.assertAllClose(result, angle)
