"""Methods for testing the Geometric Median Estimators."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.spd_matrices import SPDMetricAffine
from geomstats.learning.geometric_median import WeiszfeldAlgorithm

ROOT2 = gs.sqrt(2)

class TestGeometricMedianEstimation(geomstats.tests.TestCase):
    """Test of Geometric Meidan Estimators"""

    def test_fit_SPD_manifold(self):
        """Test the fit method on SPD manifold"""
        
        wa_estimator = WeiszfeldAlgorithm(SPDMetricAffine(n=2))
        X = gs.array([[[1., 0.], [0., 1.]],
                      [[1., 0.], [0., 1]]])
        
        wa_estimator.fit(X)
        gm_expected = gs.array([[1., 0.], [0., 1.]])
        gm_result = wa_estimator.estimate_

        self.assertAllClose(gm_expected, gm_result )


        X = gs.array([[[1., 0.], [0., 1.]],
                      [[2., 0.], [0., 2]]])

        wa_estimator.fit(X)
        gm_expected = gs.array([[ROOT2, 0.], [0., ROOT2]])
        gm_result = wa_estimator.estimate_

    