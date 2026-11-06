import scipy

import geomstats.backend as gs
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.test.test_case import TestCase


def _estimate_von_mises_kappa(X, n_steps=100):
    n_points = X.shape[0]
    dim = X.shape[1] - 1

    sum_points = gs.sum(X, axis=0)
    mean_norm = gs.linalg.norm(sum_points) / n_points

    kappa_estimate = mean_norm * (dim + 1.0 - mean_norm**2) / (1.0 - mean_norm**2)
    p = dim + 1
    for _ in range(n_steps):
        bessel_func_1 = scipy.special.iv(p / 2.0, kappa_estimate)
        bessel_func_2 = scipy.special.iv(p / 2.0 - 1.0, kappa_estimate)
        ratio = bessel_func_1 / bessel_func_2
        denominator = 1.0 - ratio**2 - (p - 1.0) * ratio / kappa_estimate
        kappa_estimate -= (ratio - mean_norm) / denominator

    return kappa_estimate


class DistributionTestCase(TestCase):
    def test_sample_belongs(self, n_samples, atol=gs.atol):
        """Check samples belong to the space.

        Also checks the batch shape of the samples, as `belongs` returns
        one boolean per point.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        atol : float
            Absolute tolerance.
        """
        sample = self.distribution.sample(n_samples)
        res = self.distribution.space.belongs(sample, atol=atol)

        expected = gs.ones(n_samples, dtype=bool) if n_samples > 1 else gs.array(True)
        self.assertAllEqual(res, expected)


class VonMisesFisherTestCase(DistributionTestCase):
    def _get_mean_direction(self):
        if self.distribution.mu is not None:
            return self.distribution.mu
        return gs.array([1.0] + [0.0] * self.distribution.space.dim)

    def test_sample_mean(self, n_samples, atol=gs.atol):
        """Check the normalized mean of the samples is the mean direction.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        atol : float
            Absolute tolerance.
        """
        sample = self.distribution.sample(n_samples)

        sum_point = gs.sum(sample, axis=0)
        res = sum_point / gs.linalg.norm(sum_point)
        self.assertAllClose(res, self._get_mean_direction(), atol=atol)

    def test_sample_kappa(self, n_samples, atol=gs.atol):
        """Check the maximum likelihood estimate of kappa.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        atol : float
            Absolute tolerance.
        """
        sample = self.distribution.sample(n_samples)

        kappa_estimate = _estimate_von_mises_kappa(sample, n_steps=100)
        self.assertAllClose(
            kappa_estimate, gs.array(self.distribution.kappa), atol=atol
        )


class HypersphereRiemannianNormalTestCase(DistributionTestCase):
    def _fit_frechet_mean(self, sample):
        space = self.distribution.space
        if space.dim == 1:
            estimator = FrechetMean(space)
        else:
            estimator = FrechetMean(space, method="adaptive")

        estimator.fit(sample)

        return estimator.estimate_

    def test_sample_frechet_mean(self, n_samples, atol=gs.atol):
        """Check the Frechet mean of the samples is the mean.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        atol : float
            Absolute tolerance.
        """
        sample = self.distribution.sample(n_samples)
        estimate = self._fit_frechet_mean(sample)

        expected = self.distribution.mean
        if expected is None:
            expected = gs.array([0.0] * self.distribution.space.dim + [1.0])

        self.assertAllClose(estimate, expected, atol=atol)
