"""Methods for testing the MDM classifier module."""
import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.spd_matrices import SPDMetricAffine
from geomstats.learning.mdm import RiemannianMinimumDistanceToMeanClassifier


e = gs.exp(1.)


class TestRiemannianMinimumDistanceToMeanClassifier(geomstats.tests.TestCase):
    """Test of Riemannian MDM classifier."""

    @geomstats.tests.np_only
    def test_fit(self):
        """Test the fit method."""
        n_clusters = 2
        MDMEstimator = RiemannianMinimumDistanceToMeanClassifier(
            SPDMetricAffine(n=2), n_clusters, point_type='matrix')

        points_A = gs.array([[[e ** 2, 0], [0, 1]],
                             [[1, 0], [0, 1]]])
        labels_A = gs.array([[1, 0],
                             [1, 0]])
        bary_A_expected = gs.array([[e, 0],
                                    [0, 1]])

        points_B = gs.array([[[e ** 8, 0], [0, 1]],
                             [[1, 0], [0, 1]]])
        labels_B = gs.array([[0, 1],
                             [0, 1]])
        bary_B_expected = gs.array([[e ** 4, 0],
                                    [0, 1]])

        train_data = gs.concatenate([points_A, points_B])
        train_labels = gs.concatenate([labels_A, labels_B])

        MDMEstimator.fit(train_data, train_labels)

        bary_A_result = MDMEstimator.mean_estimate[0]
        bary_B_result = MDMEstimator.mean_estimate[1]

        self.assertAllClose(bary_A_result, bary_A_expected)
        self.assertAllClose(bary_B_result, bary_B_expected)

    @geomstats.tests.np_only
    def test_predict(self):
        """Test the predict method."""
        n_clusters = 2
        bary_A = gs.array([[e, 0],
                           [0, 1]])
        bary_B = gs.array([[e ** 4, 0],
                           [0, 1]])

        MDMEstimator = RiemannianMinimumDistanceToMeanClassifier(
            SPDMetricAffine(n=2), n_clusters, point_type='matrix')
        MDMEstimator.mean_estimate = gs.concatenate(
            [bary_A[None, ...], bary_B[None, ...]])

        X = gs.array([[e ** 3, 0],
                      [0, 1]])[None, ...]

        # distance_AX_expected = 2.
        # distance_BX_expected = 1.
        Y_expected = gs.array([[0, 1]])

        Y_result = MDMEstimator.predict(X)

        self.assertAllClose(Y_result, Y_expected)
