"""Unit tests for the Hypersphere."""

import scipy.special

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean
from tests.conftest import Parametrizer, np_backend
from tests.data.hypersphere_data import HypersphereMetricTestData, HypersphereTestData
from tests.geometry_test_cases import LevelSetTestCase, RiemannianMetricTestCase

MEAN_ESTIMATION_TOL = 1e-1
KAPPA_ESTIMATION_TOL = 1e-1
ONLINE_KMEANS_TOL = 1e-1


class TestHypersphere(LevelSetTestCase, metaclass=Parametrizer):
    testing_data = HypersphereTestData()

    def test_replace_values(self, dim, points, new_points, indcs, expected):
        space = self.Space(dim)
        result = space._replace_values(points, new_points, indcs)
        self.assertAllClose(result, expected)

    def test_angle_to_extrinsic(self, dim, point, expected):
        space = self.Space(dim)
        result = space.angle_to_extrinsic(point)
        self.assertAllClose(result, expected)

    def test_extrinsic_to_angle(self, dim, point, expected):
        space = self.Space(dim)
        result = space.extrinsic_to_angle(point)
        self.assertAllClose(result, expected)

    def test_spherical_to_extrinsic(self, dim, point, expected):
        space = self.Space(dim)
        result = space.spherical_to_extrinsic(point)
        self.assertAllClose(result, expected)

    def test_extrinsic_to_spherical(self, dim, point, expected):
        space = self.Space(dim)
        result = space.extrinsic_to_spherical(point)
        self.assertAllClose(result, expected)

    def test_random_von_mises_fisher_belongs(self, dim, n_samples):
        space = self.Space(dim)
        result = space.belongs(space.random_von_mises_fisher(n_samples=n_samples))
        self.assertTrue(gs.all(result))

    def test_random_von_mises_fisher_mean(self, dim, kappa, n_samples, expected):
        space = self.Space(dim)
        points = space.random_von_mises_fisher(kappa=kappa, n_samples=n_samples)
        sum_points = gs.sum(points, axis=0)
        result = sum_points / gs.linalg.norm(sum_points)
        self.assertAllClose(result, expected, atol=KAPPA_ESTIMATION_TOL)

    def test_tangent_spherical_to_extrinsic(
        self, dim, tangent_vec_spherical, base_point_spherical, expected
    ):
        space = self.Space(dim)
        result = space.tangent_spherical_to_extrinsic(
            tangent_vec_spherical, base_point_spherical
        )
        self.assertAllClose(result, expected)

    def test_tangent_extrinsic_to_spherical(
        self, dim, tangent_vec, base_point, base_point_spherical, expected
    ):
        space = self.Space(dim)
        result = space.tangent_extrinsic_to_spherical(
            tangent_vec, base_point, base_point_spherical
        )
        self.assertAllClose(result, expected)

    def test_tangent_extrinsic_to_spherical_raises(
        self, dim, tangent_vec, base_point, base_point_spherical, expected
    ):
        space = self.Space(dim)
        with expected:
            space.tangent_extrinsic_to_spherical(
                tangent_vec, base_point, base_point_spherical
            )

    def test_riemannian_normal_frechet_mean(self, dim):
        space = self.Space(dim)
        mean = space.random_uniform()
        precision = gs.eye(space.dim) * 10
        sample = space.random_riemannian_normal(mean, precision, 30000)
        estimator = FrechetMean(space.metric, method="adaptive")
        estimator.fit(sample)
        estimate = estimator.estimate_
        self.assertAllClose(estimate, mean, atol=1e-1)

    def test_riemannian_normal_and_belongs(self, dim, n_points):
        space = self.Space(dim)
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
        space = self.Space(dim)
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


class HypersphereMetricTestCase(RiemannianMetricTestCase):
    def test_inner_product(
        self, space, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        space.equip_with_metric(self.Metric)
        result = space.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(result, expected)

    def test_dist(self, space, point_a, point_b, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.dist(point_a, point_b)
        self.assertAllClose(result, expected)

    def test_dist_pairwise(self, space, point, expected, rtol):
        space.equip_with_metric(self.Metric)
        result = space.metric.dist_pairwise(point)
        self.assertAllClose(result, expected, rtol=rtol)

    def test_diameter(self, space, points, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.diameter(points)
        self.assertAllClose(result, expected)

    def test_christoffels_shape(self, space, point, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.christoffels(point)
        self.assertAllClose(gs.shape(result), expected)

    def test_riemann_tensor_spherical_coords_shape(self, space, base_point, expected):
        """Test the shape of the Riemann tensor on the sphere.

        Note that the base_point is input in spherical coordinates.
        """
        space.equip_with_metric(self.Metric)
        result = space.metric.riemann_tensor(base_point).shape
        self.assertAllClose(expected, result)

    def test_riemann_tensor_spherical_coords(self, space, base_point):
        """Test the Riemann tensor on the sphere.

        riemann_tensor[...,i,j,k,l] = R_{ijk}^l
            Riemannian tensor curvature,
            with the contravariant index on the last dimension.

        Note that the base_point is input in spherical coordinates.

        Expected formulas taken from:
        https://digitalcommons.latech.edu/cgi/viewcontent.cgi?
        article=1008&context=mathematics-senior-capstone-papers
        """
        space.equip_with_metric(self.Metric)
        riemann_tensor_ijk_l = space.metric.riemann_tensor(base_point)
        theta, _ = base_point[0], base_point[1]
        expected_212_1 = gs.sin(theta) ** 2
        expected_221_1 = -gs.sin(theta) ** 2
        expected_121_2 = 1
        expected_112_2 = -1
        result_212_1 = riemann_tensor_ijk_l[1, 0, 1, 0]
        result_221_1 = riemann_tensor_ijk_l[1, 1, 0, 0]
        result_121_2 = riemann_tensor_ijk_l[0, 1, 0, 1]
        result_112_2 = riemann_tensor_ijk_l[0, 0, 1, 1]
        self.assertAllClose(expected_212_1, result_212_1)
        self.assertAllClose(expected_221_1, result_221_1)
        self.assertAllClose(expected_121_2, result_121_2)
        self.assertAllClose(expected_112_2, result_112_2)

    def test_ricci_tensor_spherical_coords_shape(self, space, base_point, expected):
        """Test the shape of the Ricci tensor on the sphere.

        ricci_tensor[...,i,j] = R_{ij}
            Ricci tensor curvature.

        Note that the base_point is input in spherical coordinates.
        """
        space.equip_with_metric(self.Metric)
        result = space.metric.ricci_tensor(base_point).shape
        self.assertAllClose(expected, result)

    def test_ricci_tensor_spherical_coords(self, space, base_point, expected):
        """Test the Ricci tensor on the sphere.

        ricci_tensor[...,i,j] = R_{ij}
            Ricci tensor curvature.

        Note that the base_point is input in spherical coordinates.

        Expected formulas taken from:
        https://digitalcommons.latech.edu/cgi/viewcontent.cgi?
        article=1008&context=mathematics-senior-capstone-papers
        """
        space.equip_with_metric(self.Metric)
        result = space.metric.ricci_tensor(base_point)
        self.assertAllClose(expected, result)

    def test_sectional_curvature(
        self, space, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        space.equip_with_metric(self.Metric)
        result = space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(result, expected, atol=1e-2)

    def test_exp_and_dist_and_projection_to_tangent_space(
        self, space, vector, base_point
    ):
        space.equip_with_metric(self.Metric)
        tangent_vec = space.to_tangent(vector=vector, base_point=base_point)
        exp = space.metric.exp(tangent_vec=tangent_vec, base_point=base_point)
        result = space.metric.dist(base_point, exp)
        expected = gs.linalg.norm(tangent_vec) % (2 * gs.pi)
        self.assertAllClose(result, expected)


class TestHypersphereMetric(HypersphereMetricTestCase, metaclass=Parametrizer):
    skip_test_exp_geodesic_ivp = True
    skip_test_dist_point_to_itself_is_zero = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature = True
    skip_test_sectional_curvature_shape = True
    skip_test_riemann_tensor_spherical_coords_shape = np_backend()
    skip_test_ricci_tensor_spherical_coords_shape = True
    skip_test_riemann_tensor_spherical_coords = np_backend()
    skip_test_ricci_tensor_spherical_coords = True

    testing_data = HypersphereMetricTestData()
