"""Methods for testing the MDM classifier module."""
import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.spd_matrices import SPDMetricAffine
from geomstats.learning.mdm import RiemannianMinimumDistanceToMeanClassifier


EULER = gs.exp(1.)


class TestRiemannianMinimumDistanceToMeanClassifier(geomstats.tests.TestCase):
    """Test of Riemannian MDM classifier."""

    def test_fit(self):
        """Test the fit method."""
        n_clusters = 2
        MDMEstimator = RiemannianMinimumDistanceToMeanClassifier(
            SPDMetricAffine(n=2), n_clusters, point_type='matrix')

        points_a = gs.array([[[EULER ** 2, 0], [0, 1]],
                             [[1, 0], [0, 1]]])
        labels_a = gs.array([[1, 0],
                             [1, 0]])
        bary_a_expected = gs.array([[EULER, 0],
                                    [0, 1]])

        points_b = gs.array([[[EULER ** 8, 0], [0, 1]],
                             [[1, 0], [0, 1]]])
        labels_b = gs.array([[0, 1],
                             [0, 1]])
        bary_b_expected = gs.array([[EULER ** 4, 0],
                                    [0, 1]])

        train_data = gs.concatenate([points_a, points_b])
        train_labels = gs.concatenate([labels_a, labels_b])

        MDMEstimator.fit(train_data, train_labels)

        bary_a_result = MDMEstimator.mean_estimates_[0]
        bary_b_result = MDMEstimator.mean_estimates_[1]

        self.assertAllClose(bary_a_result, bary_a_expected)
        self.assertAllClose(bary_b_result, bary_b_expected)

    def test_predict(self):
        """Test the predict method."""
        n_clusters = 2
        bary_a = gs.array([[EULER, 0],
                           [0, 1]])
        bary_b = gs.array([[EULER ** 4, 0],
                           [0, 1]])

        MDMEstimator = RiemannianMinimumDistanceToMeanClassifier(
            SPDMetricAffine(n=2), n_clusters, point_type='matrix')
        MDMEstimator.mean_estimates_ = gs.concatenate(
            [bary_a[None, ...], bary_b[None, ...]])

        X = gs.array([[EULER ** 3, 0],
                      [0, 1]])[None, ...]

        y_expected = gs.array([[0, 1]])

        y_result = MDMEstimator.predict(X)

        self.assertAllClose(y_result, y_expected)
