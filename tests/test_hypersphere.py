"""
Unit tests for the Hypersphere.
"""

import scipy.special
import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere

MEAN_ESTIMATION_TOL = 1e-6
KAPPA_ESTIMATION_TOL = 1e-2
ONLINE_KMEANS_TOL = 2e-2


class TestHypersphereMethods(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.dimension = 4
        self.space = Hypersphere(dimension=self.dimension)
        self.metric = self.space.metric
        self.n_samples = 10

    @geomstats.tests.np_and_pytorch_only
    def test_random_uniform_and_belongs(self):
        """
        Test that the random uniform method samples
        on the hypersphere space.
        """
        n_samples = self.n_samples
        point = self.space.random_uniform(n_samples)
        result = self.space.belongs(point)
        expected = gs.array([[True]] * n_samples)

        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_pytorch_only
    def test_random_uniform(self):
        point = self.space.random_uniform()

        self.assertAllClose(gs.shape(point), (1, self.dimension + 1))

    def test_projection_and_belongs(self):
        point = gs.array([1., 2., 3., 4., 5.])
        proj = self.space.projection(point)
        result = self.space.belongs(proj)
        expected = gs.array([[True]])

        self.assertAllClose(expected, result)

    def test_intrinsic_and_extrinsic_coords(self):
        """
        Test that the composition of
        intrinsic_to_extrinsic_coords and
        extrinsic_to_intrinsic_coords
        gives the identity.
        """
        point_int = gs.array([.1, 0., 0., .1])
        point_ext = self.space.intrinsic_to_extrinsic_coords(point_int)
        result = self.space.extrinsic_to_intrinsic_coords(point_ext)
        expected = point_int
        expected = helper.to_vector(expected)

        self.assertAllClose(result, expected)

        point_ext = (1. / (gs.sqrt(6.))
                     * gs.array([1., 0., 0., 1., 2.]))
        point_int = self.space.extrinsic_to_intrinsic_coords(point_ext)
        result = self.space.intrinsic_to_extrinsic_coords(point_int)
        expected = point_ext
        expected = helper.to_vector(expected)

        self.assertAllClose(result, expected)

    def test_intrinsic_and_extrinsic_coords_vectorization(self):
        """
        Test that the composition of
        intrinsic_to_extrinsic_coords and
        extrinsic_to_intrinsic_coords
        gives the identity.
        """
        point_int = gs.array(
            [[.1, 0., 0., .1],
             [.1, .1, .1, .4],
             [.1, .3, 0., .1],
             [-0.1, .1, -.4, .1],
             [0., 0., .1, .1],
             [.1, .1, .1, .1]])
        point_ext = self.space.intrinsic_to_extrinsic_coords(point_int)
        result = self.space.extrinsic_to_intrinsic_coords(point_ext)
        expected = point_int
        expected = helper.to_vector(expected)

        self.assertAllClose(result, expected)

        point_int = self.space.extrinsic_to_intrinsic_coords(point_ext)
        result = self.space.intrinsic_to_extrinsic_coords(point_int)
        expected = point_ext
        expected = helper.to_vector(expected)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_log_and_exp_general_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.

        NB: points on the n-dimensional sphere are
        (n+1)-D vectors of norm 1.
        """
        # Riemannian Log then Riemannian Exp
        # General case
        base_point = gs.array([1., 2., 3., 4., 6.])
        base_point = base_point / gs.linalg.norm(base_point)
        point = gs.array([0., 5., 6., 2., -1.])
        point = point / gs.linalg.norm(point)

        log = self.metric.log(point=point, base_point=base_point)
        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        expected = point
        expected = helper.to_vector(expected)

        self.assertAllClose(result, expected, atol=1e-6)

    @geomstats.tests.np_and_pytorch_only
    def test_log_and_exp_edge_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.

        NB: points on the n-dimensional sphere are
        (n+1)-D vectors of norm 1.
        """
        # Riemannian Log then Riemannian Exp
        # Edge case: two very close points, base_point_2 and point_2,
        # form an angle < epsilon
        base_point = gs.array([1., 2., 3., 4., 6.])
        base_point = base_point / gs.linalg.norm(base_point)
        point = (base_point
                 + 1e-12 * gs.array([-1., -2., 1., 1., .1]))
        point = point / gs.linalg.norm(point)

        log = self.metric.log(point=point, base_point=base_point)
        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        expected = point
        expected = helper.to_vector(expected)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_exp_vectorization(self):
        n_samples = self.n_samples
        dim = self.dimension + 1

        one_vec = self.space.random_uniform()
        one_base_point = self.space.random_uniform()
        n_vecs = self.space.random_uniform(n_samples=n_samples)
        n_base_points = self.space.random_uniform(n_samples=n_samples)

        one_tangent_vec = self.space.projection_to_tangent_space(
            one_vec, base_point=one_base_point)
        result = self.metric.exp(one_tangent_vec, one_base_point)

        self.assertAllClose(gs.shape(result), (1, dim))

        n_tangent_vecs = self.space.projection_to_tangent_space(
            n_vecs, base_point=one_base_point)
        result = self.metric.exp(n_tangent_vecs, one_base_point)

        self.assertAllClose(gs.shape(result), (n_samples, dim))

        one_tangent_vec = self.space.projection_to_tangent_space(
            one_vec, base_point=n_base_points)
        result = self.metric.exp(one_tangent_vec, n_base_points)

        self.assertAllClose(gs.shape(result), (n_samples, dim))

        n_tangent_vecs = self.space.projection_to_tangent_space(
            n_vecs, base_point=n_base_points)
        result = self.metric.exp(n_tangent_vecs, n_base_points)

        self.assertAllClose(gs.shape(result), (n_samples, dim))

    @geomstats.tests.np_and_pytorch_only
    def test_log_vectorization(self):
        n_samples = self.n_samples
        dim = self.dimension + 1

        one_base_point = self.space.random_uniform()
        one_point = self.space.random_uniform()
        n_points = self.space.random_uniform(n_samples=n_samples)
        n_base_points = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.log(one_point, one_base_point)
        self.assertAllClose(gs.shape(result), (1, dim))

        result = self.metric.log(n_points, one_base_point)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        result = self.metric.log(one_point, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        result = self.metric.log(n_points, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

    @geomstats.tests.np_and_pytorch_only
    def test_exp_and_log_and_projection_to_tangent_space_general_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.

        NB: points on the n-dimensional sphere are
        (n+1)-D vectors of norm 1.
        """
        # TODO(nina): Fix that this test fails, also in numpy
        # Riemannian Exp then Riemannian Log
        # General case
        # NB: Riemannian log gives a regularized tangent vector,
        # so we take the norm modulo 2 * pi.
        base_point = gs.array([0., -3., 0., 3., 4.])
        base_point = base_point / gs.linalg.norm(base_point)
        vector = gs.array([9., 5., 0., 0., -1.])
        vector = self.space.projection_to_tangent_space(
            vector=vector, base_point=base_point)

        # exp = self.metric.exp(tangent_vec=vector, base_point=base_point)
        # result = self.metric.log(point=exp, base_point=base_point)

        expected = vector
        norm_expected = gs.linalg.norm(expected)
        regularized_norm_expected = gs.mod(norm_expected, 2 * gs.pi)
        expected = expected / norm_expected * regularized_norm_expected
        expected = helper.to_vector(expected)

    @geomstats.tests.np_and_pytorch_only
    def test_exp_and_log_and_projection_to_tangent_space_edge_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.

        NB: points on the n-dimensional sphere are
        (n+1)-D vectors of norm 1.
        """
        # Riemannian Exp then Riemannian Log
        # Edge case: tangent vector has norm < epsilon
        base_point = gs.array([10., -2., -.5, 34., 3.])
        base_point = base_point / gs.linalg.norm(base_point)
        vector = 1e-10 * gs.array([.06, -51., 6., 5., 3.])
        vector = self.space.projection_to_tangent_space(
            vector=vector, base_point=base_point)

        exp = self.metric.exp(tangent_vec=vector, base_point=base_point)
        result = self.metric.log(point=exp, base_point=base_point)
        expected = self.space.projection_to_tangent_space(
            vector=vector, base_point=base_point)
        expected = helper.to_vector(expected)

        self.assertAllClose(result, expected, atol=1e-8)

    def test_squared_norm_and_squared_dist(self):
        """
        Test that the squared distance between two points is
        the squared norm of their logarithm.
        """
        point_a = (1. / gs.sqrt(129.)
                   * gs.array([10., -2., -5., 0., 0.]))
        point_b = (1. / gs.sqrt(435.)
                   * gs.array([1., -20., -5., 0., 3.]))
        log = self.metric.log(point=point_a, base_point=point_b)
        result = self.metric.squared_norm(vector=log)
        expected = self.metric.squared_dist(point_a, point_b)
        expected = helper.to_scalar(expected)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_squared_dist_vectorization(self):
        n_samples = self.n_samples

        one_point_a = self.space.random_uniform()
        one_point_b = self.space.random_uniform()
        n_points_a = self.space.random_uniform(n_samples=n_samples)
        n_points_b = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.squared_dist(one_point_a, one_point_b)
        self.assertAllClose(gs.shape(result), (1, 1))

        result = self.metric.squared_dist(n_points_a, one_point_b)
        self.assertAllClose(gs.shape(result), (n_samples, 1))

        result = self.metric.squared_dist(one_point_a, n_points_b)
        self.assertAllClose(gs.shape(result), (n_samples, 1))

        result = self.metric.squared_dist(n_points_a, n_points_b)
        self.assertAllClose(gs.shape(result), (n_samples, 1))

    def test_norm_and_dist(self):
        """
        Test that the distance between two points is
        the norm of their logarithm.
        """
        point_a = (1. / gs.sqrt(129.)
                   * gs.array([10., -2., -5., 0., 0.]))
        point_b = (1. / gs.sqrt(435.)
                   * gs.array([1., -20., -5., 0., 3.]))
        log = self.metric.log(point=point_a, base_point=point_b)
        result = self.metric.norm(vector=log)
        expected = self.metric.dist(point_a, point_b)
        expected = helper.to_scalar(expected)

        self.assertAllClose(result, expected)

    def test_dist_point_and_itself(self):
        # Distance between a point and itself is 0
        point_a = (1. / gs.sqrt(129.)
                   * gs.array([10., -2., -5., 0., 0.]))
        point_b = point_a
        result = self.metric.dist(point_a, point_b)
        expected = 0.
        expected = helper.to_scalar(expected)

        self.assertAllClose(result, expected)

    def test_dist_orthogonal_points(self):
        # Distance between two orthogonal points is pi / 2.
        point_a = gs.array([10., -2., -.5, 0., 0.])
        point_a = point_a / gs.linalg.norm(point_a)
        point_b = gs.array([2., 10, 0., 0., 0.])
        point_b = point_b / gs.linalg.norm(point_b)
        result = gs.dot(point_a, point_b)
        result = helper.to_scalar(result)
        expected = 0
        expected = helper.to_scalar(expected)
        self.assertAllClose(result, expected)

        result = self.metric.dist(point_a, point_b)
        expected = gs.pi / 2
        expected = helper.to_scalar(expected)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_exp_and_dist_and_projection_to_tangent_space(self):
        base_point = gs.array([16., -2., -2.5, 84., 3.])
        base_point = base_point / gs.linalg.norm(base_point)
        vector = gs.array([9., 0., -1., -2., 1.])
        tangent_vec = self.space.projection_to_tangent_space(
            vector=vector, base_point=base_point)

        exp = self.metric.exp(
            tangent_vec=tangent_vec, base_point=base_point)
        result = self.metric.dist(base_point, exp)
        expected = gs.linalg.norm(tangent_vec) % (2 * gs.pi)
        expected = helper.to_scalar(expected)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_exp_and_dist_and_projection_to_tangent_space_vec(self):
        base_point = gs.array([
            [16., -2., -2.5, 84., 3.],
            [16., -2., -2.5, 84., 3.]])

        base_single_point = gs.array([16., -2., -2.5, 84., 3.])
        scalar_norm = gs.linalg.norm(base_single_point)

        base_point = base_point / scalar_norm
        vector = gs.array(
            [[9., 0., -1., -2., 1.],
             [9., 0., -1., -2., 1]])

        tangent_vec = self.space.projection_to_tangent_space(
            vector=vector, base_point=base_point)

        exp = self.metric.exp(
            tangent_vec=tangent_vec, base_point=base_point)

        result = self.metric.dist(base_point, exp)
        expected = gs.linalg.norm(tangent_vec, axis=-1) % (2 * gs.pi)

        expected = helper.to_scalar(expected)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_geodesic_and_belongs(self):
        n_geodesic_points = 100
        initial_point = self.space.random_uniform()
        vector = gs.array([2., 0., -1., -2., 1.])
        initial_tangent_vec = self.space.projection_to_tangent_space(
            vector=vector, base_point=initial_point)
        geodesic = self.metric.geodesic(
            initial_point=initial_point,
            initial_tangent_vec=initial_tangent_vec)

        t = gs.linspace(start=0., stop=1., num=n_geodesic_points)
        points = geodesic(t)

        result = self.space.belongs(points)
        expected = gs.array(n_geodesic_points * [[True]])

        self.assertAllClose(expected, result)

    def test_inner_product(self):
        tangent_vec_a = gs.array([1., 0., 0., 0., 0.])
        tangent_vec_b = gs.array([0., 1., 0., 0., 0.])
        base_point = gs.array([0., 0., 0., 0., 1.])
        result = self.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point)
        expected = gs.array([[0.]])

        self.assertAllClose(expected, result)

    def test_diameter(self):
        dim = 2
        sphere = Hypersphere(dim)
        point_a = gs.array([[0., 0., 1.]])
        point_b = gs.array([[1., 0., 0.]])
        point_c = gs.array([[0., 0., -1.]])
        result = sphere.metric.diameter(gs.vstack((point_a, point_b, point_c)))
        expected = gs.pi
        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_pytorch_only
    def test_closest_neighbor_index(self):
        """
        Check that the closest neighbor is one of neighbors.
        """
        n_samples = 10
        points = self.space.random_uniform(n_samples=n_samples)
        point = points[0, :]
        neighbors = points[1:, :]
        index = self.metric.closest_neighbor_index(point, neighbors)
        closest_neighbor = points[index, :]

        test = gs.sum(gs.all(points == closest_neighbor, axis=1))
        result = test > 0
        self.assertTrue(result)

    @geomstats.tests.np_and_pytorch_only
    def test_sample_von_mises_fisher(self):
        """
        Check that the maximum likelihood estimates of the mean and
        concentration parameter are close to the real values. A first
        estimation of the concentration parameter is obtained by a
        closed-form expression and improved through the Newton method.
        """
        dim = 2
        n_points = 1000000
        sphere = Hypersphere(dim)

        # check mean value for concentrated distribution
        kappa = 10000000
        points = sphere.random_von_mises_fisher(kappa, n_points)
        sum_points = gs.sum(points, axis=0)
        mean = gs.array([0., 0., 1.])
        mean_estimate = sum_points / gs.linalg.norm(sum_points)
        expected = mean
        result = mean_estimate
        self.assertTrue(
            gs.allclose(result, expected, atol=MEAN_ESTIMATION_TOL)
        )
        # check concentration parameter for dispersed distribution
        kappa = 1
        points = sphere.random_von_mises_fisher(kappa, n_points)
        sum_points = gs.sum(points, axis=0)
        mean_norm = gs.linalg.norm(sum_points) / n_points
        kappa_estimate = (mean_norm * (dim + 1. - mean_norm**2)
                          / (1. - mean_norm**2))
        kappa_estimate = gs.cast(kappa_estimate, gs.float64)
        p = dim + 1
        n_steps = 100
        for i in range(n_steps):
            bessel_func_1 = scipy.special.iv(p / 2., kappa_estimate)
            bessel_func_2 = scipy.special.iv(p / 2. - 1., kappa_estimate)
            ratio = bessel_func_1 / bessel_func_2
            denominator = 1. - ratio**2 - (p - 1.) * ratio / kappa_estimate
            mean_norm = gs.cast(mean_norm, gs.float64)
            kappa_estimate = kappa_estimate - (ratio - mean_norm) / denominator
        expected = kappa
        result = kappa_estimate
        self.assertTrue(
            gs.allclose(result, expected, atol=KAPPA_ESTIMATION_TOL))

    @geomstats.tests.np_and_pytorch_only
    def test_spherical_to_extrinsic(self):
        """
        Check vectorization of conversion from spherical
        to extrinsic coordinates on the 2-sphere.
        """
        dim = 2
        sphere = Hypersphere(dim)
        points_spherical = gs.array([[gs.pi / 2, 0],
                                     [gs.pi / 6, gs.pi / 4]])
        result = sphere.spherical_to_extrinsic(points_spherical)
        expected = gs.array(
            [[1., 0., 0.],
             [gs.sqrt(2) / 4, gs.sqrt(2) / 4, gs.sqrt(3) / 2]])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_tangent_spherical_to_extrinsic(self):
        """
        Check vectorization of conversion from spherical
        to extrinsic coordinates for tangent vectors to the
        2-sphere.
        """
        dim = 2
        sphere = Hypersphere(dim)
        base_points_spherical = gs.array([[gs.pi / 2, 0],
                                          [gs.pi / 2, 0]])
        tangent_vecs_spherical = gs.array([[0.25, 0.5],
                                          [0.3, 0.2]])
        result = sphere.tangent_spherical_to_extrinsic(
            tangent_vecs_spherical, base_points_spherical)
        expected = gs.array([[0, 0.5, -0.25],
                             [0, 0.2, -0.3]])
        self.assertAllClose(result, expected)

    def test_christoffels_vectorization(self):
        """
        Check vectorization of Christoffel symbols in
        spherical coordinates on the 2-sphere.
        """
        dim = 2
        sphere = Hypersphere(dim)
        points_spherical = gs.array([[gs.pi / 2, 0],
                                     [gs.pi / 6, gs.pi / 4]])
        christoffel = sphere.metric.christoffels(
            points_spherical)
        result = christoffel.shape
        expected = gs.array([2, dim, dim, dim])
        self.assertAllClose(result, expected)
