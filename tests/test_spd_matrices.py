"""Unit tests for the manifold of symmetric positive definite matrices."""

import math
import warnings

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.spd_matrices import (
    SPDMatrices,
    SPDMetricAffine,
    SPDMetricEuclidean,
    SPDMetricLogEuclidean,
    SPDMetricProcrustes
)


class TestSPDMatrices(geomstats.tests.TestCase):
    """Test of SPDMatrices methods."""

    def setUp(self):
        """Set up the test."""
        warnings.simplefilter('ignore', category=ImportWarning)

        gs.random.seed(1234)

        self.n = 3
        self.space = SPDMatrices(n=self.n)
        self.metric_affine = SPDMetricAffine(n=self.n)
        self.metric_procrustes = SPDMetricProcrustes(n=self.n)
        self.metric_euclidean = SPDMetricEuclidean(n=self.n)
        self.metric_logeuclidean = SPDMetricLogEuclidean(n=self.n)
        self.n_samples = 4

    def test_belongs(self):
        """Test of belongs method."""
        mats = gs.array(
            [[3., -1.], [-1., 3.]])
        result = SPDMatrices(2).belongs(mats)
        expected = True
        self.assertAllClose(result, expected)

        mats = gs.array(
            [[-1., -1.], [-1., 3.]])
        result = SPDMatrices(2).belongs(mats)
        expected = False
        self.assertAllClose(result, expected)

        mats = gs.eye(3)
        result = SPDMatrices(2).belongs(mats)
        expected = False
        self.assertAllClose(result, expected)

    def test_belongs_vectorization(self):
        """Test of belongs method."""
        mats = gs.array([
            [[1., 0], [0, 1.]],
            [[1., 2.], [2., 1.]],
            [[1., 0.], [1., 1.]]])
        result = SPDMatrices(2).belongs(mats)
        expected = gs.array([True, False, False])
        self.assertAllClose(result, expected)

    def test_random_uniform_and_belongs(self):
        """Test of random_uniform and belongs methods."""
        point = self.space.random_uniform()
        result = self.space.belongs(point)
        expected = True
        self.assertAllClose(result, expected)

    def test_random_uniform_and_belongs_vectorization(self):
        """Test of random_uniform and belongs methods."""
        points = self.space.random_uniform(4)
        result = self.space.belongs(points)
        expected = gs.array([True] * 4)
        self.assertAllClose(result, expected)

    def test_vector_from_symmetric_matrix_and_symmetric_matrix_from_vector(
            self):
        """Test for matrix to vector and vector to matrix conversions."""
        sym_mat_1 = gs.array([[1., 0.6, -3.],
                              [0.6, 7., 0.],
                              [-3., 0., 8.]])
        vector_1 = self.space.to_vector(sym_mat_1)
        result_1 = self.space.from_vector(vector_1)
        expected_1 = sym_mat_1

        self.assertTrue(gs.allclose(result_1, expected_1))

        vector_2 = gs.array([1., 2., 3., 4., 5., 6.])
        sym_mat_2 = self.space.from_vector(vector_2)
        result_2 = self.space.to_vector(sym_mat_2)
        expected_2 = vector_2

        self.assertTrue(gs.allclose(result_2, expected_2))

    def test_vector_and_symmetric_matrix_vectorization(self):
        """Test of vectorization."""
        n_samples = self.n_samples
        vector = gs.random.rand(n_samples, 6)
        sym_mat = self.space.from_vector(vector)
        result = self.space.to_vector(sym_mat)
        expected = vector

        self.assertTrue(gs.allclose(result, expected))

        sym_mat = self.space.random_uniform(n_samples)
        vector = self.space.to_vector(sym_mat)
        result = self.space.from_vector(vector)
        expected = sym_mat

        self.assertTrue(gs.allclose(result, expected))

    def test_logm(self):
        """Test of logm method."""
        expected = gs.array([[[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]]])
        c = math.cosh(1)
        s = math.sinh(1)
        e = math.exp(1)
        v = gs.array([[[c, s, 0.], [s, c, 0.], [0., 0., e]]])
        result = self.space.logm(v)
        self.assertAllClose(result, expected)

    def test_differential_power(self):
        """Test of differential_power method."""
        base_point = gs.array([[1., 0., 0.],
                               [0., 2.5, 1.5],
                               [0., 1.5, 2.5]])
        tangent_vec = gs.array([[2., 1., 1.],
                               [1., .5, .5],
                               [1., .5, .5]])
        power = .5
        result = self.space.differential_power(
            power=power,
            tangent_vec=tangent_vec,
            base_point=base_point)
        expected = gs.array([[1., 1 / 3, 1 / 3],
                             [1 / 3, .125, .125],
                             [1 / 3, .125, .125]])
        self.assertAllClose(result, expected)

    def test_inverse_differential_power(self):
        """Test of inverse_differential_power method."""
        base_point = gs.array([[1., 0., 0.],
                               [0., 2.5, 1.5],
                               [0., 1.5, 2.5]])
        tangent_vec = gs.array([[1., 1 / 3, 1 / 3],
                                [1 / 3, .125, .125],
                                [1 / 3, .125, .125]])
        power = .5
        result = self.space.inverse_differential_power(
            power=power,
            tangent_vec=tangent_vec,
            base_point=base_point)
        expected = gs.array([[2., 1., 1.],
                             [1., .5, .5],
                             [1., .5, .5]])
        self.assertAllClose(result, expected)

    def test_differential_log(self):
        """Test of differential_log method."""
        base_point = gs.array([[1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., 4.]])
        tangent_vec = gs.array([[1., 1., 3.],
                                [1., 1., 3.],
                                [3., 3., 4.]])
        result = self.space.differential_log(tangent_vec, base_point)
        x = 2 * gs.log(2.)
        expected = gs.array([[1., 1., x],
                             [1., 1., x],
                             [x, x, 1]])

        self.assertAllClose(result, expected)

    def test_inverse_differential_log(self):
        """Test of inverse_differential_log method."""
        base_point = gs.array([[1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., 4.]])
        x = 2 * gs.log(2.)
        tangent_vec = gs.array([[1., 1., x],
                                [1., 1., x],
                                [x, x, 1]])
        result = self.space.inverse_differential_log(tangent_vec, base_point)
        expected = gs.array([[1., 1., 3.],
                             [1., 1., 3.],
                             [3., 3., 4.]])
        self.assertAllClose(result, expected)

    def test_differential_exp(self):
        """Test of differential_exp method."""
        base_point = gs.array([[1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., -1.]])
        tangent_vec = gs.array([[1., 1., 1.],
                                [1., 1., 1.],
                                [1., 1., 1.]])
        result = self.space.differential_exp(tangent_vec, base_point)
        x = gs.exp(1.)
        y = gs.sinh(1.)
        expected = gs.array([[x, x, y],
                             [x, x, y],
                             [y, y, 1 / x]])

        self.assertAllClose(result, expected)

    def test_inverse_differential_exp(self):
        """Test of inverse_differential_exp method."""
        base_point = gs.array([[1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., -1.]])
        x = gs.exp(1.)
        y = gs.sinh(1.)
        tangent_vec = gs.array([[x, x, y],
                                [x, x, y],
                                [y, y, 1. / x]])
        result = self.space.inverse_differential_exp(tangent_vec, base_point)
        expected = gs.array([[1., 1., 1.],
                             [1., 1., 1.],
                             [1., 1., 1.]])
        self.assertAllClose(result, expected)

    def test_procrustes_inner_product(self):
        """Test of SPDMetricProcrustes.inner_product method."""
        base_point = gs.array([[1., 0., 0.],
                               [0., 1.5, .5],
                               [0., .5, 1.5]])
        tangent_vec_a = gs.array([[2., 1., 1.],
                                  [1., .5, .5],
                                  [1., .5, .5]])
        tangent_vec_b = gs.array([[1., 2., 4.],
                                  [2., 3., 8.],
                                  [4., 8., 5.]])
        metric = SPDMetricProcrustes(3)
        result = metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        expected = gs.array(4.)

        self.assertAllClose(result, expected)

    def test_power_affine_inner_product(self):
        """Test of SPDMetricAffine.inner_product method."""
        base_point = gs.array([[1., 0., 0.],
                               [0., 2.5, 1.5],
                               [0., 1.5, 2.5]])
        tangent_vec = gs.array([[2., 1., 1.],
                               [1., .5, .5],
                               [1., .5, .5]])
        metric = SPDMetricAffine(3, power_affine=.5)
        result = metric.inner_product(tangent_vec, tangent_vec, base_point)
        expected = 713 / 144

        self.assertAllClose(result, expected)

    def test_power_euclidean_inner_product(self):
        """Test of SPDMetricEuclidean.inner_product method."""
        base_point = gs.array([[1., 0., 0.],
                               [0., 2.5, 1.5],
                               [0., 1.5, 2.5]])
        tangent_vec = gs.array([[2., 1., 1.],
                                [1., .5, .5],
                                [1., .5, .5]])
        metric = SPDMetricEuclidean(3, power_euclidean=.5)
        result = metric.inner_product(tangent_vec, tangent_vec, base_point)
        expected = 3472 / 576

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_euclidean_exp_domain(self):
        """Test of SPDMetricEuclidean.exp_domain method."""
        base_point = gs.array([[1., 0., 0.],
                               [0., 2., 0.],
                               [0., 0., 3.]])
        tangent_vec = gs.array([[-1., 0., 0.],
                                [0., -.5, 0.],
                                [0., 0., 1.]])
        metric = self.metric_euclidean
        result = metric.exp_domain(tangent_vec, base_point)
        expected = gs.array([-3, 1])

        self.assertAllClose(result, expected)

    def test_log_euclidean_inner_product(self):
        """Test of SPDMetricLogEuclidean.inner_product method."""
        base_point = gs.array([[1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., 4.]])
        tangent_vec = gs.array([[1., 1., 3.],
                                [1., 1., 3.],
                                [3., 3., 4.]])
        metric = self.metric_logeuclidean
        result = metric.inner_product(tangent_vec, tangent_vec, base_point)
        x = 2 * gs.log(2.)
        expected = 5. + 4. * x ** 2

        self.assertAllClose(result, expected)

    def test_log_and_exp_affine_invariant(self):
        """Test of SPDMetricAffine.log and exp methods with power=1."""
        base_point = gs.array([[5., 0., 0.],
                               [0., 7., 2.],
                               [0., 2., 8.]])
        point = gs.array([[9., 0., 0.],
                          [0., 5., 0.],
                          [0., 0., 1.]])

        metric = self.metric_affine
        log = metric.log(point=point, base_point=base_point)
        result = metric.exp(tangent_vec=log, base_point=base_point)
        expected = point

        self.assertAllClose(result, expected)

    def test_log_and_exp_power_affine(self):
        """Test of SPDMetricAffine.log and exp methods with power!=1."""
        base_point = gs.array([[5., 0., 0.],
                               [0., 7., 2.],
                               [0., 2., 8.]])
        point = gs.array([[9., 0., 0.],
                          [0., 5., 0.],
                          [0., 0., 1.]])
        metric = SPDMetricAffine(3, power_affine=.5)
        log = metric.log(point, base_point)
        result = metric.exp(log, base_point)
        expected = point
        self.assertAllClose(result, expected)

    def test_log_and_exp_logeuclidean(self):
        """Test of SPDMetricLogEuclidean.log and exp methods."""
        base_point = gs.array([[5., 0., 0.],
                               [0., 7., 2.],
                               [0., 2., 8.]])
        point = gs.array([[9., 0., 0.],
                          [0., 5., 0.],
                          [0., 0., 1.]])

        metric = self.metric_logeuclidean
        log = metric.log(point=point, base_point=base_point)
        result = metric.exp(tangent_vec=log, base_point=base_point)
        expected = point

        self.assertAllClose(result, expected, atol=1e-5)

    def test_exp_and_belongs(self):
        """Test of SPDMetricAffine.exp with power=1 and belongs methods."""
        n_samples = self.n_samples
        base_point = self.space.random_uniform(n_samples=1)
        tangent_vec = self.space.random_tangent_vec_uniform(
            n_samples=n_samples, base_point=base_point)
        metric = self.metric_affine
        exps = metric.exp(tangent_vec, base_point)
        result = self.space.belongs(exps)
        expected = gs.array([True] * n_samples)

        self.assertAllClose(result, expected)

    def test_exp_vectorization(self):
        """Test of SPDMetricAffine.exp with power=1 and vectorization."""
        n_samples = self.n_samples
        one_base_point = self.space.random_uniform(n_samples=1)
        n_base_point = self.space.random_uniform(n_samples=n_samples)

        n_tangent_vec_same_base = self.space.random_tangent_vec_uniform(
            n_samples=n_samples, base_point=one_base_point)
        n_tangent_vec = self.space.random_tangent_vec_uniform(
            n_samples=n_samples, base_point=n_base_point)
        metric = self.metric_affine

        # Test with the 1 base_point, and several different tangent_vecs
        result = metric.exp(n_tangent_vec_same_base, one_base_point)

        self.assertAllClose(
            gs.shape(result), (n_samples, self.space.n, self.space.n))

        # Test with the same number of base_points and tangent_vecs
        result = metric.exp(n_tangent_vec, n_base_point)

        self.assertAllClose(
            gs.shape(result), (n_samples, self.space.n, self.space.n))

    def test_log_vectorization(self):
        """Test of SPDMetricAffine.log with power 1 and vectorization."""
        n_samples = self.n_samples
        one_base_point = self.space.random_uniform(n_samples=1)
        n_base_point = self.space.random_uniform(n_samples=n_samples)

        one_point = self.space.random_uniform(n_samples=1)
        n_point = self.space.random_uniform(n_samples=n_samples)
        metric = self.metric_affine

        # Test with different points, one base point
        result = metric.log(n_point, one_base_point)

        self.assertAllClose(
            gs.shape(result), (n_samples, self.space.n, self.space.n))

        # Test with the same number of points and base points
        result = metric.log(n_point, n_base_point)

        self.assertAllClose(
            gs.shape(result), (n_samples, self.space.n, self.space.n))

        # Test with the one point and n base points
        result = metric.log(one_point, n_base_point)

        self.assertAllClose(
            gs.shape(result), (n_samples, self.space.n, self.space.n))

    def test_geodesic_and_belongs(self):
        """Test of SPDMetricAffine.geodesic with power 1 and belongs."""
        initial_point = self.space.random_uniform()
        initial_tangent_vec = self.space.random_tangent_vec_uniform(
            n_samples=1, base_point=initial_point)
        metric = self.metric_affine
        geodesic = metric.geodesic(
            initial_point=initial_point,
            initial_tangent_vec=initial_tangent_vec)

        n_points = 10
        t = gs.linspace(start=0., stop=1., num=n_points)
        points = geodesic(t)
        result = self.space.belongs(points)
        expected = gs.array([True] * n_points)

        self.assertAllClose(result, expected)

    def test_squared_dist_is_symmetric(self):
        """Test of SPDMetricAffine.squared_dist (power=1) and is_symmetric."""
        n_samples = self.n_samples

        point_1 = self.space.random_uniform(n_samples=1)
        point_2 = self.space.random_uniform(n_samples=1)
        point_1 = gs.cast(point_1, gs.float64)
        point_2 = gs.cast(point_2, gs.float64)

        metric = self.metric_affine

        sq_dist_1_2 = metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = metric.squared_dist(point_2, point_1)

        self.assertAllClose(sq_dist_1_2, sq_dist_2_1, atol=1e-3)

        point_2 = self.space.random_uniform(n_samples=n_samples)
        point_2 = gs.cast(point_2, gs.float64)

        sq_dist_1_2 = metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = metric.squared_dist(point_2, point_1)
        self.assertAllClose(sq_dist_1_2, sq_dist_2_1, atol=1e-3)

        point_1 = self.space.random_uniform(n_samples=n_samples)
        point_2 = self.space.random_uniform(n_samples=1)
        point_1 = gs.cast(point_1, gs.float64)
        point_2 = gs.cast(point_2, gs.float64)

        sq_dist_1_2 = metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = metric.squared_dist(point_2, point_1)

        self.assertAllClose(sq_dist_1_2, sq_dist_2_1, atol=1e-3)

        point_1 = self.space.random_uniform(n_samples=n_samples)
        point_2 = self.space.random_uniform(n_samples=n_samples)
        point_1 = gs.cast(point_1, gs.float64)
        point_2 = gs.cast(point_2, gs.float64)

        sq_dist_1_2 = metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = metric.squared_dist(point_2, point_1)

        self.assertAllClose(sq_dist_1_2, sq_dist_2_1, atol=1e-3)

    def test_squared_dist_vectorization(self):
        """Test of SPDMetricAffine.squared_dist (power=1) and vectorization."""
        n_samples = self.n_samples
        point_1 = self.space.random_uniform(n_samples=n_samples)
        point_2 = self.space.random_uniform(n_samples=n_samples)

        metric = self.metric_affine
        result = metric.squared_dist(point_1, point_2)

        self.assertAllClose(gs.shape(result), (n_samples,))

        point_1 = self.space.random_uniform(n_samples=1)
        point_2 = self.space.random_uniform(n_samples=n_samples)

        result = metric.squared_dist(point_1, point_2)

        self.assertAllClose(gs.shape(result), (n_samples,))

        point_1 = self.space.random_uniform(n_samples=n_samples)
        point_2 = self.space.random_uniform(n_samples=1)

        result = metric.squared_dist(point_1, point_2)

        self.assertAllClose(gs.shape(result), (n_samples,))

        point_1 = self.space.random_uniform(n_samples=1)
        point_2 = self.space.random_uniform(n_samples=1)

        result = metric.squared_dist(point_1, point_2)

        self.assertAllClose(gs.shape(result), ())

    def test_parallel_transport_affine_invariant(self):
        """Test of SPDMetricAffine.parallel_transport method with power=1."""
        n_samples = self.n_samples
        gs.random.seed(1)
        point = self.space.random_uniform(n_samples)
        tan_a = self.space.random_tangent_vec_uniform(n_samples, point)
        tan_b = self.space.random_tangent_vec_uniform(n_samples, point)

        metric = self.metric_affine
        expected = metric.norm(tan_a, point)
        end_point = metric.exp(tan_b, point)

        transported = metric.parallel_transport(tan_a, tan_b, point)
        result = metric.norm(transported, end_point)

        self.assertAllClose(expected, result, atol=1e-4)
