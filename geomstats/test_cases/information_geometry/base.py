import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.vectorization import get_batch_shape


class InformationManifoldMixinTestCase(TestCase):
    def _check_sample_belongs_to_support(self, sample, atol):
        raise NotImplementedError(
            "Need to define ``_check_sample_belongs_to_support``."
        )

    def test_point_to_pdf(self, x, point, expected, atol=gs.atol):
        """Check probability density function.

        Parameters
        ----------
        x : array-like, shape=[n_samples, *support_shape]
            Points of the support where the density is evaluated.
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., n_samples]
            Expected densities.
        atol : float
            Absolute tolerance.
        """
        res = self.space.point_to_pdf(point)(x)
        self.assertAllClose(res, expected, atol=atol)

    def test_point_to_cdf(self, x, point, expected, atol=gs.atol):
        """Check cumulative distribution function.

        Parameters
        ----------
        x : array-like, shape=[n_samples, *support_shape]
            Points of the support where the distribution is evaluated.
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., n_samples]
            Expected cumulative probabilities.
        atol : float
            Absolute tolerance.
        """
        res = self.space.point_to_cdf(point)(x)
        self.assertAllClose(res, expected, atol=atol)

    def test_sample_shape(self, point, n_samples):
        """Check shape of samples.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        n_samples : int
            Number of samples.
        """
        res = self.space.sample(point, n_samples=n_samples)

        expected_shape = (
            get_batch_shape(self.space.point_ndim, point)
            + (n_samples,)
            + self.space.support_shape
        )
        self.assertEqual(res.shape, expected_shape)

    def test_sample_belongs_to_support(self, point, n_samples, atol=gs.atol):
        """Check samples belong to the support.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        n_samples : int
            Number of samples.
        atol : float
            Absolute tolerance.
        """
        sample = self.space.sample(point, n_samples)
        self._check_sample_belongs_to_support(sample, atol)

    def test_point_to_pdf_against_scipy(self, point, n_samples, atol=gs.atol):
        """Check probability density function against scipy.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        n_samples : int
            Number of samples where the density is evaluated.
        atol : float
            Absolute tolerance.

        Notes
        -----
        The samples are drawn from the first point.
        """
        sample_point = point if point.ndim == self.space.point_ndim else point[0]
        sample = self.space.sample(sample_point, n_samples=n_samples)

        res = self.space.point_to_pdf(point)(sample)
        res_ = self.random_variable.pdf(sample, point)
        self.assertAllClose(res, res_, atol=atol)

    def test_point_to_cdf_bounds(self, point, n_samples):
        """Check cumulative distribution function is in [0, 1].

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        n_samples : int
            Number of samples where the distribution is evaluated.

        Notes
        -----
        The samples are drawn from the first point.
        """
        sample_point = point if point.ndim == self.space.point_ndim else point[0]
        sample = self.space.sample(sample_point, n_samples)

        res = self.space.point_to_cdf(point)(sample)
        self.assertTrue(gs.all(res >= 0.0))
        self.assertTrue(gs.all(res <= 1.0))
