"""Methods for testing the incremental frechet mean estimator."""

import geomstats.backend as gs
from geomstats.distributions.lognormal import LogNormal
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricLogEuclidean
from geomstats.learning.incremental_frechet_mean import IncrementalFrechetMean


class TestIncrementalFrechetMean:
    def test_fit_log_euclidean():
        manifold = SPDMatrices(3, metric=SPDMetricLogEuclidean(3))
        identity = gs.eye(3)
        cov = gs.eye(6) / 6
        samples = LogNormal(manifold, identity).sample(500)
        estimator = IncrementalFrechetMean(SPDMetricLogEuclidean(3))
        estimate = estimator.fit(samples).estimate_

    # def test_fit_affine_invariant():

    # def test_fit_euclidean():
