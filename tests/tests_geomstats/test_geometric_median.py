"""Methods for testing the Geometric Median."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricAffine
from geomstats.learning.geometric_median import GeometricMedian

ROOT2 = gs.sqrt(2.0)


class TestGeometricMedian(geomstats.tests.TestCase):
    """Test of Geometric Median Estimators"""

    def test_median_for_spd_manifold(self):
        """Test the fit method on SPD manifold"""
        wa_estimator = GeometricMedian(SPDMetricAffine(n=2))
        X = gs.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])

        wa_estimator.fit(X)
        gm_expected = gs.array([[1.0, 0.0], [0.0, 1.0]])
        gm_result = wa_estimator.estimate_

        self.assertAllClose(gm_expected, gm_result)

    def test_median_belongs_to_spd_manifold(self):
        """Test the fit method on SPD manifold"""
        n = 5
        n_samples = 10
        wa_estimator = GeometricMedian(SPDMetricAffine(n))
        space = SPDMatrices(n)
        X = space.random_point(n_samples)
        gm = wa_estimator.fit(X).estimate_

        self.assertTrue(space.belongs(gm))
