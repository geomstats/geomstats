"""Unit tests for the Dirichlet manifold."""

import math
import warnings

import tests.helper as helper
from scipy.stats import dirichlet

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.information_geometry.dirichlet import DirichletDistributions
from geomstats.information_geometry.dirichlet import DirichletMetric


class TestDirichletDistributions(geomstats.tests.TestCase):
    """Class defining the Dirichlet distributions tests.
    """
    def setUp(self):
        """Define the parameters of the tests."""
        gs.random.seed(0)
        warnings.simplefilter('ignore', category=UserWarning)
        self.dim = 3
        self.dirichlet = DirichletDistributions(self.dim)
        self.metric = DirichletMetric(self.dim)
        self.n_points = 10
        self.n_samples = 20

    def test_random_uniform_and_belongs(self):
        """Test random_uniform and belongs.

        Test that the random uniform method samples
        on the Dirichlet distribution space.
        """
        point = self.dirichlet.random_point()
        result = self.dirichlet.belongs(point)
        expected = True
        self.assertAllClose(expected, result)

    def test_random_uniform_and_belongs_vectorization(self):
        """Test random_uniform and belongs.

        Test that the random uniform method samples
        on the Dirichlet distribution space.
        """
        points = self.dirichlet.random_point(self.n_points)
        result = self.dirichlet.belongs(points)
        expected = gs.array([True] * self.n_points)
        self.assertAllClose(expected, result)

    def test_random_uniform(self):
        """Test random_uniform.

        Test that the random uniform method samples points of the right shape.
        """
        points = self.dirichlet.random_point(self.n_points)
        self.assertAllClose(gs.shape(points), (self.n_points, self.dim))

    @geomstats.tests.np_only
    def test_sample(self):
        """Test sample.

        Check that the samples have the right shape.
        """
        point = self.dirichlet.random_point()
        samples = self.dirichlet.sample(point, self.n_samples)
        result = samples.shape
        expected = (self.n_samples, self.dim)
        self.assertAllClose(expected, result)

        points = self.dirichlet.random_point(self.n_points)
        samples = self.dirichlet.sample(points, self.n_samples)
        result = samples.shape
        expected = (self.n_points, self.n_samples, self.dim)
        self.assertAllClose(expected, result)

    @geomstats.tests.np_only
    def test_sample_belong(self):
        """Test that sample samples in the simplex.

        Check that samples belong to the simplex,
        i.e. that their components sum up to one.
        """
        points = self.dirichlet.random_point(self.n_points)
        samples = self.dirichlet.sample(points, self.n_samples)
        result = gs.sum(samples, -1)
        expected = gs.ones((self.n_points, self.n_samples))
        self.assertAllClose(expected, result)

    @geomstats.tests.np_only
    def test_point_to_pdf(self):
        """Test point_to_pdf.

        Check vectorization of the computation of the pdf.
        """
        n_points = 2
        points = self.dirichlet.random_point(n_points)
        pdfs = self.dirichlet.point_to_pdf(points)
        alpha = gs.ones(self.dim)
        samples = self.dirichlet.sample(alpha, self.n_samples)
        result = pdfs(samples)
        pdf1 = [dirichlet.pdf(x, points[0, :]) for x in samples]
        pdf2 = [dirichlet.pdf(x, points[1, :]) for x in samples]
        expected = gs.stack([gs.array(pdf1), gs.array(pdf2)], axis=0)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_metric_matrix_vectorization(self):
        """Test metric matrix vectorization..

        Check vectorization of the metric matrix.
        """
        points = self.dirichlet.random_point(self.n_points)
        mat = self.dirichlet.metric.metric_matrix(points)
        result = mat.shape
        expected = (self.n_points, self.dim, self.dim)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_metric_matrix_dim2(self):
        """Test metric matrix in dimension 2.

        Check the metric matrix in dimension 2.
        """
        dirichlet2 = DirichletDistributions(2)
        points = dirichlet2.random_point(self.n_points)
        result = dirichlet2.metric.metric_matrix(points)

        param_a = points[:, 0]
        param_b = points[:, 1]
        polygamma_ab = gs.polygamma(1, param_a + param_b)
        polygamma_a = gs.polygamma(1, param_a)
        polygamma_b = gs.polygamma(1, param_b)
        vector = gs.stack(
            [polygamma_a - polygamma_ab,
             - polygamma_ab,
             polygamma_b - polygamma_ab], axis=-1)
        expected = SymmetricMatrices.from_vector(vector)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_christoffels(self):
        """Test Christoffel symbols in dimension 2.

        Check the Christoffel symbols in dimension 2.
        """
        gs.random.seed(123)
        dirichlet2 = DirichletDistributions(2)
        points = dirichlet2.random_point(self.n_points)
        result = dirichlet2.metric.christoffels(points)

        def coefficients(param_a, param_b):
            """Christoffel coefficients for the beta distributions."""
            poly1a = gs.polygamma(1, param_a)
            poly2a = gs.polygamma(2, param_a)
            poly1b = gs.polygamma(1, param_b)
            poly2b = gs.polygamma(2, param_b)
            poly1ab = gs.polygamma(1, param_a + param_b)
            poly2ab = gs.polygamma(2, param_a + param_b)
            metric_det = 2 * (poly1a * poly1b - poly1ab * (poly1a + poly1b))

            c1 = (poly2a * (poly1b - poly1ab) - poly1b * poly2ab) / metric_det
            c2 = - poly1b * poly2ab / metric_det
            c3 = (poly2b * poly1ab - poly1b * poly2ab) / metric_det
            return c1, c2, c3

        param_a, param_b = points[:, 0], points[:, 1]
        c1, c2, c3 = coefficients(param_a, param_b)
        c4, c5, c6 = coefficients(param_b, param_a)
        vector_0 = gs.stack([c1, c2, c3], axis=-1)
        vector_1 = gs.stack([c6, c5, c4], axis=-1)
        gamma_0 = SymmetricMatrices.from_vector(vector_0)
        gamma_1 = SymmetricMatrices.from_vector(vector_1)
        expected = gs.stack([gamma_0, gamma_1], axis=-3)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_christoffels_vectorization(self):
        """Test Christoffel synbols.

        Check vectorization of Christoffel symbols.
        """
        n_points = 2
        points = self.dirichlet.random_point(n_points)
        christoffel_1 = self.metric.christoffels(points[0, :])
        christoffel_2 = self.metric.christoffels(points[1, :])
        christoffels = self.metric.christoffels(points)

        result = christoffels.shape
        expected = gs.array(
            [n_points, self.dim, self.dim, self.dim])
        self.assertAllClose(result, expected)

        expected = gs.stack((christoffel_1, christoffel_2), axis=0)
        self.assertAllClose(christoffels, expected)

    @geomstats.tests.np_only
    def test_exp(self):
        """Test Exp.

        Test that the Riemannian exponential at points on the planes
        xk = xj in the direction of that plane stays in the plane.
        """
        n_points = 2
        gs.random.seed(123)
        points = self.dirichlet.random_point(n_points)
        vectors = self.dirichlet.random_point(n_points)
        initial_vectors = gs.array(
            [[vec_x, vec_x, vec_x] for vec_x in vectors[:, 0]])
        base_points = gs.array(
            [[param_x, param_x, param_x] for param_x in points[:, 0]])
        result = self.metric.exp(initial_vectors, base_points)
        expected = gs.transpose(gs.tile(result[:, 0], (self.dim, 1)))
        self.assertAllClose(expected, result)

        initial_vectors[:, 2] = gs.random.rand(n_points)
        base_points[:, 2] = gs.random.rand(n_points)
        result_points = self.metric.exp(initial_vectors, base_points)
        result = gs.isclose(result_points[:, 0], result_points[:, 1]).all()
        expected = gs.array([True] * n_points)
        self.assertAllClose(expected, result)

    @geomstats.tests.np_only
    def test_log_and_exp(self):
        """Test Log and Exp.

        Test that the Riemannian exponential
        and the Riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        gs.random.seed(123)
        base_points = self.dirichlet.random_point(self.n_points)
        points = self.dirichlet.random_point(self.n_points)
        log = self.metric.log(points, base_points, n_steps=500)
        expected = points
        result = self.metric.exp(tangent_vec=log, base_point=base_points)
        self.assertAllClose(result, expected, rtol=1e-2)

    @geomstats.tests.np_only
    def test_exp_vectorization(self):
        """Test vectorization of Exp.

        Test the case with one initial point and several tangent vectors.
        """
        point = self.dirichlet.random_point()
        tangent_vec = gs.array([1., 0.5, 2.])
        n_tangent_vecs = 10
        t = gs.linspace(0., 1., n_tangent_vecs)
        tangent_vecs = gs.einsum('i,...k->...ik', t, tangent_vec)
        end_points = self.metric.exp(
            tangent_vec=tangent_vecs, base_point=point)
        result = end_points.shape
        expected = (n_tangent_vecs, self.dim)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_log_vectorization(self):
        """Test vectorization of Log.

        Test the case with several base points and one end point.
        """
        base_points = self.dirichlet.random_point(self.n_points)
        point = self.dirichlet.random_point()
        tangent_vecs = self.metric.log(
            base_point=base_points, point=point)
        result = tangent_vecs.shape
        expected = (self.n_points, self.dim)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def tests_geodesic_ivp_and_bvp(self):
        """Test geodesic intial and boundary value problems.

        Check the shape of the geodesic.
        """
        n_steps = 50
        t = gs.linspace(0., 1., n_steps)

        initial_points = self.dirichlet.random_point(self.n_points)
        initial_tangent_vecs = self.dirichlet.random_point(self.n_points)
        geodesic = self.metric._geodesic_ivp(
            initial_points, initial_tangent_vecs)
        geodesic_at_t = geodesic(t)
        result = geodesic_at_t.shape
        expected = (self.n_points, n_steps, self.dim)
        self.assertAllClose(result, expected)

        end_points = self.dirichlet.random_point(self.n_points)
        geodesic = self.metric._geodesic_bvp(initial_points, end_points)
        geodesic_at_t = geodesic(t)
        result = geodesic_at_t.shape
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_geodesic(self):
        """Test geodesic.

        Check that the norm of the velocity is constant.
        """
        initial_point = self.dirichlet.random_point()
        end_point = self.dirichlet.random_point()

        n_steps = 10000
        geod = self.metric.geodesic(
            initial_point=initial_point,
            end_point=end_point)
        t = gs.linspace(0., 1., n_steps)
        geod_at_t = geod(t)
        velocity = n_steps * (geod_at_t[1:, :] - geod_at_t[:-1, :])
        velocity_norm = self.metric.norm(velocity, geod_at_t[:-1, :])
        result = 1 / velocity_norm.min() * (
            velocity_norm.max() - velocity_norm.min())
        expected = 0.

        self.assertAllClose(expected, result, rtol=1.)

    @geomstats.tests.np_only
    def test_geodesic_vectorization(self):
        """Check vectorization of geodesic.

        Check the shape of geodesic at time t for
        different scenarios.
        """
        initial_point = self.dirichlet.random_point()
        initial_tangent_vec = self.dirichlet.random_point()
        geod = self.metric.geodesic(
            initial_point=initial_point,
            initial_tangent_vec=initial_tangent_vec)
        time = 0.5
        result = geod(time).shape
        expected = (self.dim,)
        self.assertAllClose(expected, result)

        n_vecs = 5
        n_times = 10
        initial_tangent_vecs = self.dirichlet.random_point(n_vecs)
        geod = self.metric.geodesic(
            initial_point=initial_point,
            initial_tangent_vec=initial_tangent_vecs)
        times = gs.linspace(0., 1., n_times)
        result = geod(times).shape
        expected = (n_vecs, n_times, self.dim)
        self.assertAllClose(result, expected)

        end_points = self.dirichlet.random_point(self.n_points)
        geod = self.metric.geodesic(
            initial_point=initial_point,
            end_point=end_points)
        time = 0.5
        result = geod(time).shape
        expected = (self.n_points, self.dim)
        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_pytorch_only
    def test_jacobian_christoffels(self):
        """Test jacobian of Christoffel symbols.

        Compare with autograd and check vectorization.
        """
        base_point = self.dirichlet.random_point()
        result = self.metric.jacobian_christoffels(base_point)
        self.assertAllClose(
            (self.dim, self.dim, self.dim, self.dim),
            result.shape)

        expected = gs.autograd.jacobian(
            self.metric.christoffels)(base_point)
        self.assertAllClose(expected, result)

        base_points = self.dirichlet.random_point(2)
        result = self.metric.jacobian_christoffels(base_points)
        expected = [
            self.metric.jacobian_christoffels(base_points[0, :]),
            self.metric.jacobian_christoffels(base_points[1, :])]
        expected = gs.stack(expected, 0)
        self.assertAllClose(expected, result)

    @geomstats.tests.np_only
    def test_jacobian_in_geodesic_bvp(self):
        """Test Jacobian option in geodesic bvp.

        Check that dist yields the same result with
        and without.
        """
        point_a = self.dirichlet.random_point()
        point_b = self.dirichlet.random_point()
        result = self.dirichlet.metric.dist(point_a, point_b, jacobian=True)
        expected = self.dirichlet.metric.dist(point_a, point_b)
        self.assertAllClose(expected, result)

    @geomstats.tests.np_only
    def test_geodesic_bvp_timer(self):
        """Check timer for geodesic bvp."""
        max_time = 1e-4
        gs.random.seed(123)
        point_a = self.dirichlet.random_point()
        point_b = self.dirichlet.random_point()
        result = self.dirichlet.metric.dist(
            point_a, point_b, max_time=max_time)
        expected = math.nan
        self.assertAllClose(expected, result)

    def test_projection_and_belongs(self):
        """Test projection and belongs.

        Check that result of projection belongs to the space of
        Dirichlet distributions."""
        shape = (self.n_samples, self.dim)
        result = helper.test_projection_and_belongs(self.dirichlet, shape)
        for res in result:
            self.assertTrue(res)
