"""Unit tests for the Dirichlet manifold."""

import warnings

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.beta_distributions import BetaDistributions
from geomstats.geometry.dirichlet_distributions import DirichletDistributions
from geomstats.geometry.dirichlet_distributions import DirichletMetric


class TestDirichletDistributions(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)
        self.dim = 3
        self.dirichlet = DirichletDistributions(self.dim)
        self.metric = DirichletMetric(self.dim)
        self.n_samples = 10

    def test_random_uniform_and_belongs(self):
        """Test random_uniform and belongs.

        Test that the random uniform method samples
        on the Dirichlet distribution space.
        """
        point = self.dirichlet.random_uniform()
        result = self.dirichlet.belongs(point)
        expected = True
        self.assertAllClose(expected, result)

    def test_random_uniform_and_belongs_vectorization(self):
        """Test random_uniform and belongs.

        Test that the random uniform method samples
        on the Dirichlet distribution space.
        """
        n_samples = self.n_samples
        point = self.dirichlet.random_uniform(n_samples)
        result = self.dirichlet.belongs(point)
        expected = gs.array([True] * n_samples)
        self.assertAllClose(expected, result)

    def test_random_uniform(self):
        """Test random_uniform.

        Test that the random uniform method samples points of the right shape.
        """
        point = self.dirichlet.random_uniform(self.n_samples)
        self.assertAllClose(gs.shape(point), (self.n_samples, self.dim))

    def test_point_to_pdf(self):
        """Test point_to_pdf.

        Check vectorization of point_to_pdf.
        """
        points = self.dirichlet.random_uniform(self.n_samples)
        pdfs = self.dirichlet.point_to_pdf(points)
        alpha = gs.ones(self.dim)
        random_point_in_simplex = self.dirichlet.sample(alpha)
        pdfs_at_point = pdfs(random_point_in_simplex)
        self.assertAllClose(gs.shape(pdfs_at_point), (self.n_samples))

    @geomstats.tests.np_and_pytorch_only
    def test_metric_matrix_vectorization(self):
        """Test metric matrix vectorization..

        Check vectorization of the metric matrix.
        """
        points = self.dirichlet.random_uniform(self.n_samples)
        mat = self.dirichlet.metric.metric_matrix(points)
        result = mat.shape
        expected = (self.n_samples, self.dim, self.dim)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_metric_matrix_dim2(self):
        """Test metric matrix in dimension 2.

        Check that it coincides in dimension 2 with metric matrix of
        corresponding beta distributions.
        """
        dirichlet2 = DirichletDistributions(2)
        beta = BetaDistributions()
        points = dirichlet2.random_uniform(self.n_samples)
        result = dirichlet2.metric.metric_matrix(points)
        expected = beta.metric.metric_matrix(points)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_christoffels(self):
        """Test Christoffel symbols in dimension 2.

        Check that they coincide with the Christoffel symbols given by
        the beta distribution.
        """
        dirichlet2 = DirichletDistributions(2)
        beta = BetaDistributions()
        points = dirichlet2.random_uniform(self.n_samples)
        result = dirichlet2.metric.christoffels(points)
        expected = beta.metric.christoffels(points)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_christoffels_vectorization(self):
        """Test Christoffel synbols.

        Check vectorization of Christoffel symbols.
        """
        n_samples = 2
        points = self.dirichlet.random_uniform(n_samples)
        christoffel_1 = self.metric.christoffels(points[0, :])
        christoffel_2 = self.metric.christoffels(points[1, :])
        christoffels = self.metric.christoffels(points)

        result = christoffels.shape
        expected = gs.array(
            [n_samples, self.dim, self.dim, self.dim])
        self.assertAllClose(result, expected)

        expected = gs.stack((christoffel_1, christoffel_2), axis=0)
        self.assertAllClose(christoffels, expected)

    @geomstats.tests.np_only
    def test_exp(self):
        """Test Exp.

        Test that the Riemannian exponential at points on the planes
        xk = xj in the direction of that plane stays in the plane.
        """
        n_samples = 2
        gs.random.seed(123)
        points = self.dirichlet.random_uniform(n_samples)
        vectors = self.dirichlet.random_uniform(n_samples)
        initial_vectors = gs.array(
            [[vec_x, vec_x, vec_x] for vec_x in vectors[:, 0]])
        base_points = gs.array(
            [[param_x, param_x, param_x] for param_x in points[:, 0]])
        result = self.metric.exp(initial_vectors, base_points)
        expected = gs.transpose(gs.tile(result[:, 0], (self.dim, 1)))
        self.assertAllClose(expected, result)

        initial_vectors[:, 2] = gs.random.rand(n_samples)
        base_points[:, 2] = gs.random.rand(n_samples)
        result_points = self.metric.exp(initial_vectors, base_points)
        result = gs.isclose(result_points[:, 0], result_points[:, 1]).all()
        expected = gs.array([True] * n_samples)
        self.assertAllClose(expected, result)

    @geomstats.tests.np_only
    def test_log_and_exp(self):
        """Test Log and Exp.

        Test that the Riemannian exponential
        and the Riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        n_samples = self.n_samples
        gs.random.seed(123)
        base_point = self.dirichlet.random_uniform(n_samples)
        point = self.dirichlet.random_uniform(n_samples)
        log = self.metric.log(point, base_point, n_steps=500)
        expected = point
        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        self.assertAllClose(result, expected, rtol=1e-2)

    @geomstats.tests.np_only
    def test_exp_vectorization(self):
        """Test vectorization of Exp.

        Test the case with one initial point and several tangent vectors.
        """
        point = self.dirichlet.random_uniform()
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
        base_points = self.dirichlet.random_uniform(self.n_samples)
        point = self.dirichlet.random_uniform()
        tangent_vecs = self.metric.log(
            base_point=base_points, point=point)
        result = tangent_vecs.shape
        expected = (self.n_samples, self.dim)
        self.assertAllClose(result, expected)
