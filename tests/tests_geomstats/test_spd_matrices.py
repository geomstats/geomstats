"""Unit tests for the manifold of symmetric positive definite matrices."""


import math
import warnings

import pytest
from numpy.linalg import cholesky

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatrices,
)
from geomstats.geometry.spd_matrices import (
    SPDMatrices,
    SPDMetricAffine,
    SPDMetricBuresWasserstein,
    SPDMetricEuclidean,
    SPDMetricLogEuclidean,
)

SQRT_2 = math.sqrt(2)


def belongs_data():
    smoke_data = [
        dict(n=2, mat=[[3.0, -1.0], [-1.0, 3.0]], expected=True),
        dict(n=2, mat=[[1.0, 1.0], [2.0, 1.0]], expected=False),
        dict(
            n=3,
            mat=[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
            expected=True,
        ),
        dict(
            n=2,
            mat=[[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [0.0, 1.0]]],
            expected=[True, False],
        ),
    ]
    return generate_tests(smoke_data)


class TestSPDMatrices(geomstats.tests.TestCase):
    """Test of SPDMatrices methods."""

    @pytest.mark.parametrize("n, mat, expected", belongs_data())
    def test_belongs(self, n, mat, expected):
        """Test of belongs method."""
        self.assertAllClose(SPDMatrices(n).belongs(gs.array(mat)), gs.array(expected))

    @pytest.mark.parametrize("n, num_points", random_point_data())
    def test_random_point(self, n, num_points):
        """Test of random_point and belongs methods."""
        space = SPDMatrices(n)
        self.assertAllClose(
            space.belongs(space.random_point(num_points)), gs.array([True] * num_points)
        )

    @pytest.mark.parametrize("n, spd_mat, logm", logm_data())
    def test_logm(self, n, spd_mat, logm):
        """Test of logm method."""
        self.assertAllClose(SPDMatrices(n).logm(spd_mat), logm)

    @pytest.mark.parametrize("n, spd_mat, cf", cholesky_factor_data())
    def test_cholesky_factor(self, n, spd_mat, cf):
        """Test cholesky factor method"""
        result_cf = SPDMatrices(n).cholesky_factor(gs.array(spd_mat))

        self.assertAllClose(result_cf, gs.array(cf))
        self.assertAllClose(
            gs.all(PositiveLowerTriangularMatrices(n).belongs(result_cf)),
            gs.array(True),
        )

    def test_cholesky_factor_differential(self):
        """Test differential of cholesky factor map"""
        P = gs.array([[4.0, 2.0], [2.0, 5.0]])
        W = gs.array([[1.0, 1.0], [1.0, 1.0]])
        diff_chol_expected = gs.array([[1 / 4, 0.0], [3 / 8, 1 / 16]])
        diff_chol_result = self.space.differential_cholesky_factor(W, P)
        self.assertAllClose(diff_chol_expected, diff_chol_result)

    def test_cholesky_factor_differential_belongs(self):
        """Test differential of cholesky factor map for batch of inputs"""
        n_samples = 5
        P = self.space.random_point(n_samples)
        W = self.space.ambient_space.random_point(n_samples)
        diff_chol_fact_result = self.space.differential_cholesky_factor(W, P)
        belongs_expected = True
        belongs_result = gs.all(
            LowerTriangularMatrices(self.n).belongs(diff_chol_fact_result)
        )

        shape_expected = n_samples
        shape_result = diff_chol_fact_result.shape[0]

        self.assertAllClose(shape_expected, shape_result)
        self.assertAllClose(belongs_expected, belongs_result)

    def test_differential_power(self):
        """Test of differential_power method."""
        base_point = gs.array([[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]])
        tangent_vec = gs.array([[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]])
        power = 0.5
        result = self.space.differential_power(
            power=power, tangent_vec=tangent_vec, base_point=base_point
        )
        expected = gs.array(
            [[1.0, 1 / 3, 1 / 3], [1 / 3, 0.125, 0.125], [1 / 3, 0.125, 0.125]]
        )
        self.assertAllClose(result, expected)

    def test_inverse_differential_power(self):
        """Test of inverse_differential_power method."""
        base_point = gs.array([[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]])
        tangent_vec = gs.array(
            [[1.0, 1 / 3, 1 / 3], [1 / 3, 0.125, 0.125], [1 / 3, 0.125, 0.125]]
        )
        power = 0.5
        result = self.space.inverse_differential_power(
            power=power, tangent_vec=tangent_vec, base_point=base_point
        )
        expected = gs.array([[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]])
        self.assertAllClose(result, expected)

    def test_differential_log(self):
        """Test of differential_log method."""
        base_point = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]])
        tangent_vec = gs.array([[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]])
        result = self.space.differential_log(tangent_vec, base_point)
        x = 2 * gs.log(2.0)
        expected = gs.array([[1.0, 1.0, x], [1.0, 1.0, x], [x, x, 1]])

        self.assertAllClose(result, expected)

    def test_inverse_differential_log(self):
        """Test of inverse_differential_log method."""
        base_point = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]])
        x = 2 * gs.log(2.0)
        tangent_vec = gs.array([[1.0, 1.0, x], [1.0, 1.0, x], [x, x, 1]])
        result = self.space.inverse_differential_log(tangent_vec, base_point)
        expected = gs.array([[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]])
        self.assertAllClose(result, expected)

    def test_differential_exp(self):
        """Test of differential_exp method."""
        base_point = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        tangent_vec = gs.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        result = self.space.differential_exp(tangent_vec, base_point)
        x = gs.exp(1.0)
        y = gs.sinh(1.0)
        expected = gs.array([[x, x, y], [x, x, y], [y, y, 1 / x]])

        self.assertAllClose(result, expected)

    def test_inverse_differential_exp(self):
        """Test of inverse_differential_exp method."""
        base_point = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        x = gs.exp(1.0)
        y = gs.sinh(1.0)
        tangent_vec = gs.array([[x, x, y], [x, x, y], [y, y, 1.0 / x]])
        result = self.space.inverse_differential_exp(tangent_vec, base_point)
        expected = gs.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        self.assertAllClose(result, expected)

    def test_bureswasserstein_inner_product(self):
        """Test of SPDMetricBuresWasserstein.inner_product method."""
        base_point = gs.array([[1.0, 0.0, 0.0], [0.0, 1.5, 0.5], [0.0, 0.5, 1.5]])
        tangent_vec_a = gs.array([[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]])
        tangent_vec_b = gs.array([[1.0, 2.0, 4.0], [2.0, 3.0, 8.0], [4.0, 8.0, 5.0]])
        metric = SPDMetricBuresWasserstein(3)
        result = metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        expected = gs.array(4.0)

        self.assertAllClose(result, expected)

    def test_power_affine_inner_product(self):
        """Test of SPDMetricAffine.inner_product method."""
        base_point = gs.array([[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]])
        tangent_vec = gs.array([[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]])
        metric = SPDMetricAffine(3, power_affine=0.5)
        result = metric.inner_product(tangent_vec, tangent_vec, base_point)
        expected = 713 / 144

        self.assertAllClose(result, expected)

    def test_power_euclidean_inner_product(self):
        """Test of SPDMetricEuclidean.inner_product method."""
        base_point = gs.array([[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]])
        tangent_vec = gs.array([[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]])
        metric = SPDMetricEuclidean(3, power_euclidean=0.5)
        result = metric.inner_product(tangent_vec, tangent_vec, base_point)
        expected = 3472 / 576
        self.assertAllClose(result, expected)

        result = self.metric_euclidean.inner_product(
            tangent_vec, tangent_vec, base_point
        )
        expected = MatricesMetric(3, 3).inner_product(tangent_vec, tangent_vec)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_tf_only
    def test_euclidean_exp_domain(self):
        """Test of SPDMetricEuclidean.exp_domain method."""
        base_point = gs.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        tangent_vec = gs.array([[-1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 1.0]])
        metric = self.metric_euclidean
        result = metric.exp_domain(tangent_vec, base_point)
        expected = gs.array([-3, 1])

        self.assertAllClose(result, expected)

    def test_log_euclidean_inner_product(self):
        """Test of SPDMetricLogEuclidean.inner_product method."""
        base_point = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]])
        tangent_vec = gs.array([[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]])
        metric = self.metric_logeuclidean
        result = metric.inner_product(tangent_vec, tangent_vec, base_point)
        x = 2 * gs.log(2.0)
        expected = 5.0 + 4.0 * x ** 2

        self.assertAllClose(result, expected)

    def test_log_and_exp_affine_invariant(self):
        """Test of SPDMetricAffine.log and exp methods with power=1."""
        base_point = gs.array([[5.0, 0.0, 0.0], [0.0, 7.0, 2.0], [0.0, 2.0, 8.0]])
        point = gs.array([[9.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]])

        metric = self.metric_affine
        log = metric.log(point=point, base_point=base_point)
        result = metric.exp(tangent_vec=log, base_point=base_point)
        expected = point

        self.assertAllClose(result, expected)

    def test_log_and_exp_power_affine(self):
        """Test of SPDMetricAffine.log and exp methods with power!=1."""
        base_point = gs.array([[5.0, 0.0, 0.0], [0.0, 7.0, 2.0], [0.0, 2.0, 8.0]])
        point = gs.array([[9.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]])
        metric = SPDMetricAffine(3, power_affine=0.5)
        log = metric.log(point, base_point)
        result = metric.exp(log, base_point)
        expected = point
        self.assertAllClose(result, expected)

    def test_log_and_exp_bureswasserstein(self):
        """Test of SPDMetricBuresWasserstein.log and exp methods."""
        base_point = gs.array([[5.0, 0.0, 0.0], [0.0, 7.0, 2.0], [0.0, 2.0, 8.0]])
        point = gs.array([[9.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]])

        metric = self.metric_bureswasserstein
        log = metric.log(point=point, base_point=base_point)
        result = metric.exp(tangent_vec=log, base_point=base_point)
        expected = point

        self.assertAllClose(result, expected)

    def test_log_and_exp_logeuclidean(self):
        """Test of SPDMetricLogEuclidean.log and exp methods."""
        base_point = gs.array([[5.0, 0.0, 0.0], [0.0, 7.0, 2.0], [0.0, 2.0, 8.0]])
        point = gs.array([[9.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]])

        metric = self.metric_logeuclidean
        log = metric.log(point=point, base_point=base_point)
        result = metric.exp(tangent_vec=log, base_point=base_point)
        expected = point

        self.assertAllClose(result, expected)

    def test_log_and_exp_euclidean_p1(self):
        """Test of SPDMetricEuclidean.log and exp methods for power_euclidean=1."""
        base_point = gs.array([[5.0, 0.0, 0.0], [0.0, 7.0, 2.0], [0.0, 2.0, 8.0]])
        point = gs.array([[9.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]])

        metric = SPDMetricEuclidean(3, power_euclidean=1)
        log = metric.log(point=point, base_point=base_point)
        result = metric.exp(tangent_vec=log, base_point=base_point)
        expected = point

        self.assertAllClose(result, expected)

    def test_log_and_exp_euclidean_p05(self):
        """Test of SPDMetricEuclidean.log and exp methods for power_euclidean=0.5."""
        base_point = gs.array([[5.0, 0.0, 0.0], [0.0, 7.0, 2.0], [0.0, 2.0, 8.0]])
        point = gs.array([[9.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]])

        metric = SPDMetricEuclidean(3, power_euclidean=0.5)
        log = metric.log(point=point, base_point=base_point)
        result = metric.exp(tangent_vec=log, base_point=base_point)
        expected = point

        self.assertAllClose(result, expected)

    def test_exp_and_belongs(self):
        """Test of SPDMetricAffine.exp with power=1 and belongs methods."""
        n_samples = self.n_samples
        base_point = self.space.random_point(n_samples=1)
        tangent_vec = self.space.random_tangent_vec(
            n_samples=n_samples, base_point=base_point
        )
        metric = self.metric_affine
        exps = metric.exp(tangent_vec, base_point)
        result = self.space.belongs(exps)
        expected = gs.array([True] * n_samples)

        self.assertAllClose(result, expected)

    def test_exp_vectorization(self):
        """Test of SPDMetricAffine.exp with power=1 and vectorization."""
        n_samples = self.n_samples
        one_base_point = self.space.random_point(n_samples=1)
        n_base_point = self.space.random_point(n_samples=n_samples)

        n_tangent_vec_same_base = self.space.random_tangent_vec(
            n_samples=n_samples, base_point=one_base_point
        )
        n_tangent_vec = self.space.random_tangent_vec(
            n_samples=n_samples, base_point=n_base_point
        )
        metric = self.metric_affine

        # Test with the 1 base_point, and several different tangent_vecs
        result = metric.exp(n_tangent_vec_same_base, one_base_point)

        self.assertAllClose(gs.shape(result), (n_samples, self.space.n, self.space.n))

        # Test with the same number of base_points and tangent_vecs
        result = metric.exp(n_tangent_vec, n_base_point)

        self.assertAllClose(gs.shape(result), (n_samples, self.space.n, self.space.n))

    def test_log_vectorization(self):
        """Test of SPDMetricAffine.log with power 1 and vectorization."""
        n_samples = self.n_samples
        one_base_point = self.space.random_point(n_samples=1)
        n_base_point = self.space.random_point(n_samples=n_samples)

        one_point = self.space.random_point(n_samples=1)
        n_point = self.space.random_point(n_samples=n_samples)
        metric = self.metric_affine

        # Test with different points, one base point
        result = metric.log(n_point, one_base_point)

        self.assertAllClose(gs.shape(result), (n_samples, self.space.n, self.space.n))

        # Test with the same number of points and base points
        result = metric.log(n_point, n_base_point)

        self.assertAllClose(gs.shape(result), (n_samples, self.space.n, self.space.n))

        # Test with the one point and n base points
        result = metric.log(one_point, n_base_point)

        self.assertAllClose(gs.shape(result), (n_samples, self.space.n, self.space.n))

    def test_geodesic_and_belongs(self):
        """Test of SPDMetricAffine.geodesic with power 1 and belongs."""
        initial_point = self.space.random_point()
        initial_tangent_vec = self.space.random_tangent_vec(
            n_samples=1, base_point=initial_point
        )
        metric = self.metric_affine
        geodesic = metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )

        n_points = 10
        t = gs.linspace(start=0.0, stop=1.0, num=n_points)
        points = geodesic(t)
        result = self.space.belongs(points)
        self.assertTrue(gs.all(result))

    def test_squared_dist_is_symmetric(self):
        """Test of SPDMetricAffine.squared_dist (power=1) and is_symmetric."""
        n_samples = self.n_samples

        point_1 = self.space.random_point(n_samples=1)
        point_2 = self.space.random_point(n_samples=1)

        metric = self.metric_affine

        sq_dist_1_2 = metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = metric.squared_dist(point_2, point_1)

        self.assertAllClose(sq_dist_1_2, sq_dist_2_1)

        point_2 = self.space.random_point(n_samples=n_samples)

        sq_dist_1_2 = metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = metric.squared_dist(point_2, point_1)
        self.assertAllClose(sq_dist_1_2, sq_dist_2_1)

        point_1 = self.space.random_point(n_samples=n_samples)
        point_2 = self.space.random_point(n_samples=1)

        sq_dist_1_2 = metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = metric.squared_dist(point_2, point_1)

        self.assertAllClose(sq_dist_1_2, sq_dist_2_1)

        sq_dist_1_2 = metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = metric.squared_dist(point_2, point_1)

        self.assertAllClose(sq_dist_1_2, sq_dist_2_1)

    def test_squared_dist_vectorization(self):
        """Test of SPDMetricAffine.squared_dist (power=1) and vectorization."""
        n_samples = self.n_samples
        point_1 = self.space.random_point(n_samples=n_samples)
        point_2 = self.space.random_point(n_samples=n_samples)

        metric = self.metric_affine
        result = metric.squared_dist(point_1, point_2)

        self.assertAllClose(gs.shape(result), (n_samples,))

        point_1 = self.space.random_point(n_samples=1)
        point_2 = self.space.random_point(n_samples=n_samples)

        result = metric.squared_dist(point_1, point_2)

        self.assertAllClose(gs.shape(result), (n_samples,))

        point_1 = self.space.random_point(n_samples=n_samples)
        point_2 = self.space.random_point(n_samples=1)

        result = metric.squared_dist(point_1, point_2)

        self.assertAllClose(gs.shape(result), (n_samples,))

        point_1 = self.space.random_point(n_samples=1)
        point_2 = self.space.random_point(n_samples=1)

        result = metric.squared_dist(point_1, point_2)

        self.assertAllClose(gs.shape(result), ())

    def test_parallel_transport_affine_invariant(self):
        """Test of SPDMetricAffine.parallel_transport method with power=1."""
        n_samples = self.n_samples
        gs.random.seed(1)
        point = self.space.random_point(n_samples)
        tan_a = self.space.random_tangent_vec(n_samples, point)
        tan_b = self.space.random_tangent_vec(n_samples, point)

        metric = self.metric_affine
        expected = metric.norm(tan_a, point)
        end_point = metric.exp(tan_b, point)

        transported = metric.parallel_transport(tan_a, tan_b, point)
        result = metric.norm(transported, end_point)

        self.assertAllClose(expected, result)

    def test_squared_dist_bureswasserstein(self):
        """Test of SPDMetricBuresWasserstein.squared_dist method."""
        point_a = gs.array([[5.0, 0.0, 0.0], [0.0, 7.0, 2.0], [0.0, 2.0, 8.0]])
        point_b = gs.array([[9.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]])

        metric = self.metric_bureswasserstein
        result = metric.squared_dist(point_a, point_b)

        log = metric.log(point=point_b, base_point=point_a)
        expected = metric.squared_norm(vector=log, base_point=point_a)

        self.assertAllClose(result, expected)

    def test_squared_dist_bureswasserstein_vectorization(self):
        """Test of SPDMetricBuresWasserstein.squared_dist method."""
        point_a = self.space.random_point(2)
        point_b = gs.array([[9.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]])

        metric = self.metric_bureswasserstein
        result = metric.squared_dist(point_a, point_b)

        log = metric.log(point=point_b, base_point=point_a)
        expected = metric.squared_norm(vector=log, base_point=point_a)

        self.assertAllClose(result, expected)

    def test_to_tangent_and_is_tangent(self):
        mat = gs.random.rand(3, 3)
        projection = self.space.to_tangent(mat)
        result = self.space.is_tangent(projection)
        self.assertTrue(result)

    def test_projection_and_belongs(self):
        shape = (2, self.n, self.n)
        space = self.space
        result = helper.test_projection_and_belongs(space, shape)
        for res in result:
            self.assertTrue(res)
