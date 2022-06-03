"""Methods for testing the Geometric Median Estimators."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricAffine
from geomstats.learning.geometric_median import WeiszfeldAlgorithm

ROOT2 = gs.sqrt(2)


class TestGeometricMedianEstimation(geomstats.tests.TestCase):
    """Test of Geometric Meidan Estimators"""

    def test_median_for_SPD_manifold(self):
        """Test the fit method on SPD manifold"""
        wa_estimator = WeiszfeldAlgorithm(SPDMetricAffine(n=2))
        X = gs.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1]]])

        wa_estimator.fit(X)
        gm_expected = gs.array([[1.0, 0.0], [0.0, 1.0]])
        gm_result = wa_estimator.estimate_

        self.assertAllClose(gm_expected, gm_result)

    def test_median_belongs_to_SPD_manifold(self):
        """Test the fit method on SPD manifold"""
        n = 5
        n_samples = 10
        wa_estimator = WeiszfeldAlgorithm(SPDMetricAffine(n))
        SPDmanifold = SPDMatrices(n)
        X = SPDmanifold.random_point(n_samples)
        gm = wa_estimator.fit(X).estimate_

        gm_result = SPDmanifold.belongs(gm)
        gm_expected = True
        self.assertAllClose(gm_expected, gm_result)
