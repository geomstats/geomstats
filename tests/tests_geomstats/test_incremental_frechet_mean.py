"""Methods for testing the incremental frechet mean estimator."""

import geomstats.backend as gs
from geomstats.distributions.lognormal import LogNormal
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.spd_matrices import (
    SPDAffineMetric,
    SPDLogEuclideanMetric,
    SPDMatrices,
)
from geomstats.learning.incremental_frechet_mean import IncrementalFrechetMean
from tests.conftest import TestCase


class TestIncrementalFrechetMean(TestCase):
    def setup_method(self):
        """Set up  the test"""
        self.n = 3
        self.spd_cov_n = (self.n * (self.n + 1)) // 2
        self.samples = 5
        self.spd = SPDMatrices(self.n, equip=False)

        self.spd_log_euclidean = SPDMatrices(self.n, equip=False)
        self.spd_log_euclidean.equip_with_metric(SPDLogEuclideanMetric)

        self.spd_affine_invariant = SPDMatrices(self.n, equip=False)
        self.spd_affine_invariant.equip_with_metric(SPDAffineMetric)

        self.euclidean = Euclidean(self.n)

    def test_ifm_log_euclidean_belongs(self):
        mean = 2 * gs.eye(self.n)
        cov = gs.eye(self.spd_cov_n)

        LogNormalSampler = LogNormal(self.spd_log_euclidean, mean, cov)
        data = LogNormalSampler.sample(20)
        ifm = IncrementalFrechetMean(self.spd_log_euclidean).fit(data)
        ifm_mean = ifm.estimate_
        result = gs.all(self.spd.belongs(ifm_mean))
        expected = gs.array(True)
        self.assertAllClose(result, expected)

    def test_ifm_affine_invariant_belongs(self):
        mean = 2 * gs.eye(self.n)
        cov = gs.eye(self.spd_cov_n)

        LogNormalSampler = LogNormal(self.spd_affine_invariant, mean, cov)
        data = LogNormalSampler.sample(20)
        ifm = IncrementalFrechetMean(self.spd_affine_invariant).fit(data)
        ifm_mean = ifm.estimate_
        result = gs.all(self.spd.belongs(ifm_mean))
        expected = gs.array(True)
        self.assertAllClose(result, expected)

    def test_fit_euclidean(self):
        mean = gs.eye(3)
        ifm = IncrementalFrechetMean(self.euclidean).fit(mean)
        result = ifm.estimate_
        expected = gs.array([1.0, 1.0, 1.0]) / 3.0
        self.assertAllClose(result, expected)
