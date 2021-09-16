"""Methods for testing the Geometric Median Estimators."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.spd_matrices import SPDMetricAffine
from geomstats.learning.geometric_median import WeiszfeldAlgorithm





class TestRiemannianMinimumDistanceToMeanClassifier(geomstats.tests.TestCase):
    """Test of Geometric Meidan Estimators"""

    def test_fit(self):
        """Test the fit method."""
        n_clusters = 2
        wa_estimator = WeiszfeldAlgorithm(SPDMetricAffine(n=2))

        X = gs.array([[1, 0], [0, 1]],
                     [[1, 0], [0, 1]])
        
        wa_estimator.fit(X)
        gm_expected = gs.array([[1, 0], [0, 1]])
        gm_result = wa_estimator.estimate_

        self.assertAllClose(gm_expected, gm_result )