"""Unit tests for the LogNormal Sampler."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.euclidean import  Euclidean
from geomstats.sampling.lognormal import LogNormal


class TestLogNormal(geomstats.tests.TestCase):
    """Class defining the LogNormal tests."""

    def setUp(self):
        """Set up  the test"""
        self.n = 3
        self.spd_cov_n = (self.n * (self.n + 1)//2)
        self.samples = 5
        self.SPDManifold = SPDMatrices(self.n)
        self.Euclidean = Euclidean(self.n)

    def test_euclidean_belongs(self):
        """Test if the samples belong to Euclidean Space"""
        mean = gs.zeros(self.n)
        cov  = gs.eye(self.n)
        LogNormalSampler = LogNormal(self.Euclidean, mean, cov)
        data = LogNormalSampler.sample(self.samples)
        self.Eucludean.belongs(data)

    def test_spd_belongs(self):
        """Test if the samples to SPD Manifold"""
        mean = 2*gs.eye(self.n)
        cov  = gs.eye(self.spd_cov_n)
        LogNormalSampler = LogNormal(self.SPDManifold, mean, cov)
        data = LogNormalSampler.sample(self.samples)
        self.SPDManifold.belongs(data)

    def test_euclidean_frechet_mean(self):
        """Test if the frechet mean of the samples is close to mean"""            
        pass

    def test_spd_frechet_mean(self):
        """Test if the frechet mean of the samples is close to mean"""

    def test_exception_handling(self):
        """Test if the exceptions are raised for invalid parameters"""




