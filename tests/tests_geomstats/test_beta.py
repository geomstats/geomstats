"""Unit tests for the beta manifold."""

import warnings

from scipy.stats import beta

import geomstats.backend as gs
import geomstats.tests
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.information_geometry.beta import BetaMetric


class TestBetaDistributions(geomstats.tests.TestCase):
    """Class defining the beta distributions tests.
    """
    def setUp(self):
        """Define the parameters of the tests."""
        warnings.simplefilter('ignore', category=UserWarning)
        self.beta = BetaDistributions()
        self.metric = BetaMetric()
        self.n_samples = 10
        self.dim = self.beta.dim

    def test_random_uniform_and_belongs(self):
        """Test random_uniform and belongs.

        Test that the random uniform method samples
        on the beta distribution space.
        """
        point = self.beta.random_point()
        result = self.beta.belongs(point)
        expected = True
        self.assertAllClose(expected, result)

    def test_random_uniform_and_belongs_vectorization(self):
        """Test random_uniform and belongs.

        Test that the random uniform method samples
        on the beta distribution space.
        """
        n_samples = self.n_samples
        point = self.beta.random_point(n_samples)
        result = self.beta.belongs(point)
        expected = gs.array([True] * n_samples)
        self.assertAllClose(expected, result)

    def test_random_uniform(self):
        """Test random_uniform.

        Test that the random uniform method samples points of the right shape
        """
        point = self.beta.random_point(self.n_samples)
        self.assertAllClose(gs.shape(point), (self.n_samples, self.dim))

    def test_sample(self):
        """Test samples.

        Test that the sample method samples variates from beta distributions
        with the specified parameters, using the law of large numbers
        """
        n_samples = self.n_samples
        tol = (n_samples * 10) ** (- 0.5)
        point = self.beta.random_point(n_samples)
        samples = self.beta.sample(point, n_samples * 10)
        result = gs.mean(samples, axis=1)
        expected = point[:, 0] / gs.sum(point, axis=1)
        self.assertAllClose(result, expected, rtol=tol, atol=tol)

    def test_maximum_likelihood_fit(self):
        """Test maximum likelihood.

        Test that the maximum likelihood fit method recovers
        parameters of beta distribution.
        """
        n_samples = self.n_samples
        point = self.beta.random_point(n_samples)
        samples = self.beta.sample(point, n_samples * 10)
        fits = self.beta.maximum_likelihood_fit(samples)
        expected = self.beta.belongs(fits)
        result = gs.array([True] * n_samples)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_exp(self):
        """Test Exp.

        Test that the Riemannian exponential at points on the first
        bisector computed in the direction of the first bisector stays
        on the first bisector.
        """
        gs.random.seed(123)
        n_samples = self.n_samples
        points = self.beta.random_point(n_samples)
        vectors = self.beta.random_point(n_samples)
        initial_vectors = gs.array([[vec_x, vec_x] for vec_x in vectors[:, 0]])
        points = gs.array([[param_a, param_a] for param_a in points[:, 0]])
        result_points = self.metric.exp(initial_vectors, points)
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
        base_point = self.beta.random_point(n_samples=n_samples, bound=5)
        point = self.beta.random_point(n_samples=n_samples, bound=5)
        log = self.metric.log(point, base_point, n_steps=500)
        expected = point
        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        self.assertAllClose(result, expected, rtol=1e-2)

    @geomstats.tests.np_only
    def test_exp_vectorization(self):
        """Test vectorization of Exp.

        Test the case with one initial point and several tangent vectors.
        """
        point = self.beta.random_point()
        tangent_vec = gs.array([1., 2.])
        n_tangent_vecs = 10
        t = gs.linspace(0., 1., n_tangent_vecs)
        tangent_vecs = gs.einsum('i,...k->...ik', t, tangent_vec)
        end_points = self.metric.exp(
            tangent_vec=tangent_vecs, base_point=point)
        result = end_points.shape
        expected = (n_tangent_vecs, 2)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_log_vectorization(self):
        """Test vectorization of Log.

        Test the case with several base points and one end point.
        """
        n_points = 10
        base_points = self.beta.random_point(n_samples=n_points)
        point = self.beta.random_point()
        tangent_vecs = self.metric.log(
            base_point=base_points, point=point)
        result = tangent_vecs.shape
        expected = (n_points, 2)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_christoffels_vectorization(self):
        """Test Christoffel synbols.

        Check vectorization of Christoffel symbols.
        """
        points = self.beta.random_point(self.n_samples)
        christoffel = self.metric.christoffels(points)
        result = christoffel.shape
        expected = gs.array(
            [self.n_samples, self.dim, self.dim, self.dim])
        self.assertAllClose(result, expected)

    def test_metric_matrix(self):
        """Test metric matrix.

        Check the value of the metric matrix for a particular
        point in the space of beta distributions."""
        point = gs.array([1., 1.])
        result = self.beta.metric.metric_matrix(point)
        expected = gs.array([[1., -0.644934066], [-0.644934066, 1.]])
        self.assertAllClose(result, expected)
        self.assertRaises(ValueError, self.beta.metric.metric_matrix)

    def test_point_to_pdf(self):
        """Test point_to_pdf.

        Check vectorization of the computation of the pdf.
        """
        point = self.beta.random_point(n_samples=2)
        pdf = self.beta.point_to_pdf(point)
        x = gs.linspace(0., 1., 10)
        result = pdf(x)
        pdf1 = beta.pdf(x, a=point[0, 0], b=point[0, 1])
        pdf2 = beta.pdf(x, a=point[1, 0], b=point[1, 1])
        expected = gs.stack([gs.array(pdf1), gs.array(pdf2)], axis=1)
        self.assertAllClose(result, expected)
