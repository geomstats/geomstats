"""Unit tests for Frechet mean."""

import math

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.discrete_curves import DiscreteCurves, ElasticMetric
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.stiefel import Stiefel
from geomstats.learning.frechet_mean import FrechetMean, GradientDescent, variance


class TestFrechetMean(tests.conftest.TestCase):
    _multiprocess_can_split_ = True

    def setup_method(self):
        gs.random.seed(123)
        self.sphere = Hypersphere(dim=4)
        self.hyperbolic = Hyperboloid(dim=3)
        self.euclidean = Euclidean(dim=2)
        self.minkowski = Minkowski(dim=2)
        self.so3 = SpecialOrthogonal(n=3, point_type="vector")
        self.so_matrix = SpecialOrthogonal(n=3)
        self.curves_2d = DiscreteCurves(Euclidean(dim=2))
        self.curves_2d_many_sampling_points = DiscreteCurves(
            Euclidean(dim=2), k_sampling_points=300
        )

        self.curves_2d_with_elastic = DiscreteCurves(
            ambient_manifold=Euclidean(dim=2), equip=False
        )
        self.curves_2d_with_elastic.equip_with_metric(ElasticMetric, a=1, b=1)

        self.curves_2d_many_sampling_points_with_elastic = DiscreteCurves(
            Euclidean(dim=2), k_sampling_points=300
        )
        self.curves_2d_many_sampling_points_with_elastic.equip_with_metric(
            ElasticMetric, a=1, b=1
        )

    def test_logs_at_mean_curves_2d(self):
        n_tests = 10
        space = self.curves_2d
        estimator = FrechetMean(space)

        result = []
        for _ in range(n_tests):
            # take 2 random points, compute their mean, and verify that
            # log of each at the mean is opposite
            points = self.curves_2d.random_point(n_samples=2)
            estimator.fit(points)
            mean = estimator.estimate_

            # Expand and tile are added because of GitHub issue 1575
            mean = gs.expand_dims(mean, axis=0)
            mean = gs.tile(mean, (2, 1, 1))

            logs = space.metric.log(point=points, base_point=mean)
            logs_srv = space.metric.tangent_diffeomorphism(logs, base_point=mean)
            # Note that the logs are NOT inverse, only the logs_srv are.
            result.append(gs.linalg.norm(logs_srv[1, :] + logs_srv[0, :]))
        result = gs.stack(result)
        expected = gs.zeros(n_tests)
        self.assertAllClose(expected, result, atol=1e-5)

    def test_logs_at_mean_default_gradient_descent_sphere(self):
        n_tests = 10
        space = self.sphere
        estimator = FrechetMean(space, method="default")

        result = []
        for _ in range(n_tests):
            # take 2 random points, compute their mean, and verify that
            # log of each at the mean is opposite
            points = space.random_uniform(n_samples=2)
            estimator.fit(points)
            mean = estimator.estimate_

            logs = space.metric.log(point=points, base_point=mean)
            result.append(gs.linalg.norm(logs[1, :] + logs[0, :]))
        result = gs.stack(result)
        expected = gs.zeros(n_tests)
        self.assertAllClose(expected, result)

    def test_logs_at_mean_adaptive_gradient_descent_sphere(self):
        n_tests = 10
        estimator = FrechetMean(self.sphere, method="adaptive")

        result = []
        for _ in range(n_tests):
            # take 2 random points, compute their mean, and verify that
            # log of each at the mean is opposite
            points = self.sphere.random_uniform(n_samples=2)
            estimator.fit(points)
            mean = estimator.estimate_

            logs = self.sphere.metric.log(point=points, base_point=mean)
            result.append(gs.linalg.norm(logs[1, :] + logs[0, :]))
        result = gs.stack(result)

        expected = gs.zeros(n_tests)
        self.assertAllClose(expected, result)

    def test_estimate_shape_default_gradient_descent_sphere(self):
        dim = 5
        point_a = gs.array([1.0, 0.0, 0.0, 0.0, 0.0])
        point_b = gs.array([0.0, 1.0, 0.0, 0.0, 0.0])
        points = gs.array([point_a, point_b])

        mean = FrechetMean(self.sphere, method="default").set(verbose=True)
        mean.fit(points)
        result = mean.estimate_

        self.assertAllClose(gs.shape(result), (dim,))

    def test_estimate_shape_adaptive_gradient_descent_sphere(self):
        dim = 5
        point_a = gs.array([1.0, 0.0, 0.0, 0.0, 0.0])
        point_b = gs.array([0.0, 1.0, 0.0, 0.0, 0.0])
        points = gs.array([point_a, point_b])

        mean = FrechetMean(self.sphere, method="adaptive")
        mean.fit(points)
        result = mean.estimate_

        self.assertAllClose(gs.shape(result), (dim,))

    def test_estimate_shape_elastic_metric(self):
        points = self.curves_2d_with_elastic.random_point(n_samples=2)

        mean = FrechetMean(self.curves_2d_with_elastic)
        mean.fit(points)
        result = mean.estimate_

        self.assertAllClose(gs.shape(result), (points.shape[1:]))

    def test_estimate_shape_srv_metric_many_sampling_points(self):
        points = self.curves_2d_many_sampling_points.random_point(n_samples=1000)

        mean = FrechetMean(self.curves_2d_many_sampling_points)
        mean.fit(points)
        result = mean.estimate_

        self.assertAllClose(gs.shape(result), (points.shape[1:]))

    def test_estimate_shape_elastic_metric_many_sampling_points(self):
        space = self.curves_2d_many_sampling_points_with_elastic
        points = space.random_point(n_samples=1000)

        mean = FrechetMean(space)
        mean.fit(points)
        result = mean.estimate_

        self.assertAllClose(gs.shape(result), (points.shape[1:]))

    def test_estimate_shape_metric(self):
        points = self.curves_2d.random_point(n_samples=2)

        mean = FrechetMean(self.curves_2d)
        mean.fit(points)
        result = mean.estimate_

        self.assertAllClose(gs.shape(result), (points.shape[1:]))

    def test_estimate_and_belongs_default_gradient_descent_sphere(self):
        point_a = gs.array([1.0, 0.0, 0.0, 0.0, 0.0])
        point_b = gs.array([0.0, 1.0, 0.0, 0.0, 0.0])
        points = gs.array([point_a, point_b])

        mean = FrechetMean(self.sphere, method="default")
        mean.fit(points)

        result = self.sphere.belongs(mean.estimate_)
        expected = True
        self.assertAllClose(result, expected)

    def test_estimate_and_belongs_curves_2d(self):
        points = self.curves_2d.random_point(n_samples=2)

        mean = FrechetMean(self.curves_2d)
        mean.fit(points)

        result = self.curves_2d.belongs(mean.estimate_)
        expected = True
        self.assertAllClose(result, expected)

    def test_estimate_default_gradient_descent_so3(self):
        points = self.so3.random_uniform(2)

        mean_vec = FrechetMean(self.so3, method="default")
        mean_vec.fit(points)

        logs = self.so3.metric.log(points, mean_vec.estimate_)
        result = gs.sum(logs, axis=0)
        expected = gs.zeros_like(points[0])
        self.assertAllClose(result, expected)

    def test_estimate_and_belongs_default_gradient_descent_so3(self):
        point = self.so3.random_uniform(10)

        mean_vec = FrechetMean(self.so3, method="default")
        mean_vec.fit(point)

        result = self.so3.belongs(mean_vec.estimate_)
        expected = True
        self.assertAllClose(result, expected)

    @tests.conftest.np_and_autograd_only
    def test_estimate_default_gradient_descent_so_matrix(self):
        points = self.so_matrix.random_uniform(2)
        mean_vec = FrechetMean(
            self.so_matrix,
            method="default",
        )
        mean_vec.fit(points)
        logs = self.so_matrix.metric.log(points, mean_vec.estimate_)
        result = gs.sum(logs, axis=0)
        expected = gs.zeros_like(points[0])

        self.assertAllClose(result, expected, atol=1e-5)

    @tests.conftest.np_and_autograd_only
    def test_estimate_and_belongs_default_gradient_descent_so_matrix(self):
        point = self.so_matrix.random_uniform(10)

        mean = FrechetMean(self.so_matrix, method="default")
        mean.fit(point)

        result = self.so_matrix.belongs(mean.estimate_)
        expected = True
        self.assertAllClose(result, expected)

    @tests.conftest.np_and_autograd_only
    def test_estimate_and_belongs_adaptive_gradient_descent_so_matrix(self):
        point = self.so_matrix.random_uniform(10)

        mean = FrechetMean(
            self.so_matrix,
            method="adaptive",
        ).set(init_step_size=0.5, verbose=True)
        mean.fit(point)

        result = self.so_matrix.belongs(mean.estimate_)
        self.assertTrue(result)

    @tests.conftest.np_and_autograd_only
    def test_estimate_and_coincide_default_so_vec_and_mat(self):
        point = self.so_matrix.random_uniform(3)

        mean = FrechetMean(self.so_matrix, method="default")
        mean.fit(point)
        expected = mean.estimate_

        mean_vec = FrechetMean(self.so3, method="default")
        point_vec = self.so3.rotation_vector_from_matrix(point)
        mean_vec.fit(point_vec)
        result_vec = mean_vec.estimate_
        result = self.so3.matrix_from_rotation_vector(result_vec)

        self.assertAllClose(result, expected)

    def test_estimate_and_belongs_adaptive_gradient_descent_sphere(self):
        point_a = gs.array([1.0, 0.0, 0.0, 0.0, 0.0])
        point_b = gs.array([0.0, 1.0, 0.0, 0.0, 0.0])
        points = gs.array([point_a, point_b])

        mean = FrechetMean(self.sphere, method="adaptive")
        mean.fit(points)

        result = self.sphere.belongs(mean.estimate_)
        expected = True
        self.assertAllClose(result, expected)

    def test_variance_sphere(self):
        point = gs.array([0.0, 0.0, 0.0, 0.0, 1.0])
        points = gs.array([point, point])

        result = variance(self.sphere, points, base_point=point)
        expected = gs.array(0.0)

        self.assertAllClose(expected, result)

    def test_estimate_default_gradient_descent_sphere(self):
        point = gs.array([0.0, 0.0, 0.0, 0.0, 1.0])
        points = gs.array([point, point])

        mean = FrechetMean(self.sphere, method="default")
        mean.fit(X=points)

        result = mean.estimate_
        expected = point

        self.assertAllClose(expected, result)

    def test_estimate_elastic_metric(self):
        point = self.curves_2d_with_elastic.random_point(n_samples=1)
        points = gs.array([point, point])

        mean = FrechetMean(self.curves_2d_with_elastic)
        mean.fit(X=points)

        result = mean.estimate_
        expected = point

        self.assertAllClose(expected, result)

    def test_estimate_curves_2d(self):
        point = self.curves_2d.random_point(n_samples=1)
        points = gs.array([point, point])

        mean = FrechetMean(self.curves_2d)
        mean.fit(X=points)

        result = mean.estimate_
        expected = point

        self.assertAllClose(expected, result)

    def test_estimate_adaptive_gradient_descent_sphere(self):
        point = gs.array([0.0, 0.0, 0.0, 0.0, 1.0])
        points = gs.array([point, point])

        mean = FrechetMean(self.sphere, method="adaptive")
        mean.fit(X=points)

        result = mean.estimate_
        expected = point

        self.assertAllClose(expected, result)

    def test_estimate_spd(self):
        space = SPDMatrices(3)
        point = space.random_point()
        points = gs.array([point, point])
        mean = FrechetMean(space)
        mean.fit(X=points)
        result = mean.estimate_
        expected = point
        self.assertAllClose(expected, result)

    def test_estimate_spd_two_samples(self):
        space = SPDMatrices(3)
        point = space.random_point(2)
        mean = FrechetMean(space)
        mean.fit(point)
        result = mean.estimate_
        expected = space.metric.exp(space.metric.log(point[0], point[1]) / 2, point[1])
        self.assertAllClose(expected, result)

    def test_variance_hyperbolic(self):
        point = gs.array([2.0, 1.0, 1.0, 1.0])
        points = gs.array([point, point])
        result = variance(self.hyperbolic, points, base_point=point)
        expected = gs.array(0.0)

        self.assertAllClose(result, expected)

    def test_estimate_hyperbolic(self):
        point = gs.array([2.0, 1.0, 1.0, 1.0])
        points = gs.array([point, point])

        mean = FrechetMean(self.hyperbolic)
        mean.fit(X=points)
        expected = point

        result = mean.estimate_

        self.assertAllClose(result, expected)

    def test_estimate_and_belongs_hyperbolic(self):
        point_a = self.hyperbolic.random_point()
        point_b = self.hyperbolic.random_point()
        point_c = self.hyperbolic.random_point()
        points = gs.stack([point_a, point_b, point_c], axis=0)

        mean = FrechetMean(self.hyperbolic)
        mean.fit(X=points)

        result = self.hyperbolic.belongs(mean.estimate_)
        expected = True

        self.assertAllClose(result, expected)

    def test_mean_euclidean_shape(self):
        dim = 2
        point = gs.array([1.0, 4.0])

        mean = FrechetMean(self.euclidean)
        points = [point, point, point]
        mean.fit(points)

        result = mean.estimate_

        self.assertAllClose(gs.shape(result), (dim,))

    def test_mean_euclidean(self):
        point = gs.array([1.0, 4.0])

        mean = FrechetMean(self.euclidean)
        points = [point, point, point]
        mean.fit(points)

        result = mean.estimate_
        expected = point

        self.assertAllClose(result, expected)

        points = gs.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        weights = gs.array([1.0, 2.0, 1.0, 2.0])

        mean = FrechetMean(self.euclidean)
        mean.fit(points, weights=weights)

        result = mean.estimate_
        expected = gs.array([16.0 / 6.0, 22.0 / 6.0])

        self.assertAllClose(result, expected)

    def test_variance_euclidean(self):
        points = gs.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        weights = gs.array([1.0, 2.0, 1.0, 2.0])
        base_point = gs.zeros(2)
        result = variance(
            self.euclidean,
            points,
            weights=weights,
            base_point=base_point,
        )
        # we expect the average of the points' sq norms.
        expected = gs.array((1 * 5.0 + 2 * 13.0 + 1 * 25.0 + 2 * 41.0) / 6.0)

        self.assertAllClose(result, expected)

    def test_mean_matrices_shape(self):
        m, n = (2, 2)
        point = gs.array([[1.0, 4.0], [2.0, 3.0]])

        space = Matrices(m, n)
        mean = FrechetMean(space)
        points = [point, point, point]
        mean.fit(points)

        result = mean.estimate_

        self.assertAllClose(gs.shape(result), (m, n))

    def test_mean_matrices(self):
        m, n = (2, 2)
        point = gs.array([[1.0, 4.0], [2.0, 3.0]])

        space = Matrices(m, n)
        mean = FrechetMean(space)
        points = [point, point, point]
        mean.fit(points)

        result = mean.estimate_
        expected = point

        self.assertAllClose(result, expected)

    def test_mean_minkowski_shape(self):
        dim = 2
        point = gs.array([2.0, -math.sqrt(3)])
        points = [point, point, point]

        mean = FrechetMean(self.minkowski)
        mean.fit(points)
        result = mean.estimate_

        self.assertAllClose(gs.shape(result), (dim,))

    def test_mean_minkowski(self):
        point = gs.array([2.0, -math.sqrt(3)])
        points = [point, point, point]

        mean = FrechetMean(self.minkowski)
        mean.fit(points)
        result = mean.estimate_

        expected = point

        self.assertAllClose(result, expected)

        points = gs.array(
            [[1.0, 0.0], [2.0, math.sqrt(3)], [3.0, math.sqrt(8)], [4.0, math.sqrt(24)]]
        )
        weights = gs.array([1.0, 2.0, 1.0, 2.0])

        mean = FrechetMean(self.minkowski)
        mean.fit(points, weights=weights)
        result = self.minkowski.belongs(mean.estimate_)

        self.assertTrue(result)

    def test_variance_minkowski(self):
        points = gs.array(
            [[1.0, 0.0], [2.0, math.sqrt(3)], [3.0, math.sqrt(8)], [4.0, math.sqrt(24)]]
        )
        weights = gs.array([1.0, 2.0, 1.0, 2.0])
        base_point = gs.array([-1.0, 0.0])
        var = variance(
            self.minkowski,
            points,
            weights=weights,
            base_point=base_point,
        )
        result = var != 0
        # we expect the average of the points' Minkowski sq norms.
        expected = True
        self.assertAllClose(result, expected)

    def test_one_point(self):
        point = gs.array([[0.0, 0.0, 0.0, 0.0, 1.0]])

        mean = FrechetMean(self.sphere, method="default")
        mean.fit(X=point)

        result = mean.estimate_
        expected = point[0]
        self.assertAllClose(expected, result)

        mean = FrechetMean(self.sphere, method="default")
        mean.fit(X=point)

        result = mean.estimate_
        expected = point[0]
        self.assertAllClose(expected, result)

    def test_batched(self):
        space = SPDMatrices(3)
        point = space.random_point(4)
        mean_batch = FrechetMean(space, method="batch").set(verbose=True)
        data = gs.stack([point[:2], point[2:]], axis=1)
        mean_batch.fit(data)
        result = mean_batch.estimate_

        mean = FrechetMean(space)
        mean.fit(data[:, 0])
        expected_1 = mean.estimate_
        mean.fit(data[:, 1])
        expected_2 = mean.estimate_
        expected = gs.stack([expected_1, expected_2])
        self.assertAllClose(expected, result)

    @tests.conftest.np_and_autograd_only
    def test_stiefel_two_samples(self):
        space = Stiefel(3, 2)
        point = space.random_point(2)
        mean = FrechetMean(space)
        mean.fit(point)
        result = mean.estimate_
        expected = space.metric.exp(space.metric.log(point[0], point[1]) / 2, point[1])
        self.assertAllClose(expected, result)

    @tests.conftest.np_and_autograd_only
    def test_stiefel_n_samples(self):
        space = Stiefel(3, 2)
        point = space.random_point(2)
        mean = FrechetMean(space, method="default").set(
            init_step_size=0.5, verbose=True
        )
        mean.fit(point)
        result = space.belongs(mean.estimate_, atol=1e-8)
        self.assertTrue(result)

    def test_circle_mean(self):
        space = Hypersphere(1)
        points = space.random_uniform(10)
        estimator = FrechetMean(space)
        estimator.fit(points)
        mean = estimator.estimate_

        mean_gd = GradientDescent().minimize(space, points=points, weights=None)

        sum_sd_mean = gs.sum(space.metric.dist(points, mean) ** 2)
        sum_sd_mean_gd = gs.sum(space.metric.dist(points, mean_gd) ** 2)

        msg = f"circular mean: {mean}, {sum_sd_mean}\ngd: {mean_gd}, {sum_sd_mean_gd}"
        self.assertTrue(sum_sd_mean < sum_sd_mean_gd + 1e-4, msg)
