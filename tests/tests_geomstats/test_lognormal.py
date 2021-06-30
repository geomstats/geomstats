"""Unit tests for the LogNormal Sampler."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.distributions.lognormal import LogNormal
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import (
    SPDMatrices,
    SPDMetricAffine,
    SPDMetricLogEuclidean,
)


class TestLogNormal(geomstats.tests.TestCase):
    """Class defining the LogNormal tests."""

    def setUp(self):
        """Set up  the test"""
        self.n = 3
        self.spd_cov_n = (self.n * (self.n + 1)) // 2
        self.samples = 5
        self.SPDManifold = SPDMatrices(self.n)
        self.log_euclidean = SPDMetricLogEuclidean(self.n)
        self.affine_invariant = SPDMetricAffine(self.n)
        self.Euclidean = Euclidean(self.n)

    def test_euclidean_belongs(self):
        """Test if the samples belong to Euclidean Space"""
        mean = gs.zeros(self.n)
        cov = gs.eye(self.n)
        LogNormalSampler = LogNormal(self.Euclidean, mean, cov)
        data = LogNormalSampler.sample(self.samples)

        result = gs.all(self.Euclidean.belongs(data))
        expected = True
        self.assertAllClose(result, expected)

    def test_spd_belongs(self):
        """Test if the samples belong to SPD Manifold"""
        mean = 2 * gs.eye(self.n)
        cov = gs.eye(self.spd_cov_n)

        self.SPDManifold.metric = self.log_euclidean
        LogNormalSampler = LogNormal(self.SPDManifold, mean, cov)
        data = LogNormalSampler.sample(self.samples)
        result = gs.all(self.SPDManifold.belongs(data))
        expected = True
        self.assertAllClose(result, expected)

        self.SPDManifold.metric = self.affine_invariant
        LogNormalSampler = LogNormal(self.SPDManifold, mean, cov)
        data = LogNormalSampler.sample(self.samples)
        result = gs.all(self.SPDManifold.belongs(data))
        expected = True
        self.assertAllClose(result, expected)

    def test_euclidean_frechet_mean(self):
        """Test if the frechet mean of the samples is close to mean"""
        mean = gs.zeros(self.n)
        cov = gs.eye(self.n)
        data = LogNormal(self.Euclidean, mean, cov).sample(5000)
        log_data = gs.log(data)
        fm = gs.mean(log_data, axis=0)

        expected = mean
        result = fm
        self.assertAllClose(result, expected, atol=5 * 1e-2)

    def test_spd_frechet_mean(self):
        """Test if the frechet mean of the samples is close to mean"""
        mean = gs.eye(self.n)
        cov = gs.eye(self.spd_cov_n)
        data = LogNormal(self.SPDManifold, mean, cov).sample(5000)
        _fm = gs.mean(self.SPDManifold.logm(data), axis=0)
        fm = self.SPDManifold.expm(_fm)

        expected = mean
        result = fm
        self.assertAllClose(result, expected, atol=5 * 1e-2)

    def test_error_handling(self):
        """Test if the erros are raised for invalid parameters"""
        eu_mean = gs.zeros(self.n)
        spd_mean = gs.eye(self.n)
        invalid_eu_mean = gs.zeros(self.n + 1)
        invalid_spd_mean = gs.zeros((self.n, self.n))
        invalid_cov = gs.eye(self.n + 1)
        invalid_manifold = Hypersphere(dim=2)

        with self.assertRaises(ValueError):
            LogNormal(invalid_manifold, eu_mean)
        with self.assertRaises(ValueError):
            LogNormal(self.Euclidean, invalid_eu_mean)
        with self.assertRaises(ValueError):
            LogNormal(self.SPDManifold, invalid_spd_mean)
        with self.assertRaises(ValueError):
            LogNormal(self.Euclidean, eu_mean, invalid_cov)
        with self.assertRaises(ValueError):
            LogNormal(self.SPDManifold, spd_mean, invalid_cov)
