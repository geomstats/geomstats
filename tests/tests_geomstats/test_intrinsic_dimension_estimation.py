"""Methods for testing the Intrinsic Dimension Estimators"""

from sklearn import datasets
import geomstats.backend as gs
import geomstats.tests
from geomstats.learning.intrinsic_dimension_estimation import LevinaBickelEstimator


class TestIntrinsicDimensionEstimator(geomstats.tests.TestCase):
    """Test of IntrinsicDimensionEstimators."""

    def test_fit(self):
        """Test the fit method"""

        lbe = LevinaBickelEstimator()
        X = gs.random.rand(500, 3)
        dist_mat = lbe.fit(X).log_sorted_dist
        result = (dist_mat.shape[0], dist_mat.shape[1])
        expected = (500, 499)
        self.assertAllClose(result, expected)

    def test_predict(self):
        """Test the predict method."""

        X, _ = datasets.make_swiss_roll(n_samples=2000)
        lbe = LevinaBickelEstimator(min_neighbors=10, max_neighbors=200)
        int_dim = lbe.fit(X).predict()
        result = (1.9 <= int_dim) and (int_dim <= 2.1)
        expected = True
        self.assertAllClose(result, expected)
