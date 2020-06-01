import geomstats.backend as gs
import numpy as np
import geomstats.tests
from geomstats.learning.mdm import RiemannianMinimumDistanceToMeanClassifier
from geomstats.geometry.spd_matrices import SPDMetricAffine


class TestRiemannianMinimumDistanceToMeanClassifier(geomstats.tests.TestCase):
    '''
    Test of Riemannian MDM classifier
    '''

    @geomstats.tests.np_only
    def test_fit(self):

        MDMEstimator = RiemannianMinimumDistanceToMeanClassifier(
            SPDMetricAffine(n=2), point_type='matrix')

        points_A = gs.array([[[np.e**2, 0], [0, 1]],
                             [[1, 0], [0, 1]]])
        labels_A = gs.array([[1, 0],
                             [1, 0]])
        bary_A_expected = gs.array([[np.e, 0],
                                    [0, 1]])

        points_B = gs.array([[[np.e**8, 0], [0, 1]],
                             [[1, 0], [0, 1]]])
        labels_B = gs.array([[0, 1],
                             [0, 1]])
        bary_B_expected = gs.array([[np.e**4, 0],
                                    [0, 1]])

        train_data = gs.concatenate([points_A, points_B])
        train_labels = gs.concatenate([labels_A, labels_B])

        MDMEstimator.fit(train_data, train_labels)

        bary_A_result = MDMEstimator.G[0]
        bary_B_result = MDMEstimator.G[1]

        self.assertAllClose(bary_A_result, bary_A_expected)
        self.assertAllClose(bary_B_result, bary_B_expected)

        return

    @geomstats.tests.np_only
    def test_predict(self):

        bary_A = gs.array([[np.e, 0],
                           [0, 1]])
        bary_B = gs.array([[np.e**4, 0],
                           [0, 1]])

        MDMEstimator = RiemannianMinimumDistanceToMeanClassifier(
            SPDMetricAffine(n=2), point_type='matrix')
        MDMEstimator.G = gs.concatenate([bary_A[None, ...], bary_B[None, ...]])

        X = gs.array([[np.e**3, 0],
                      [0, 1]])[None, ...]

        distance_AX_expected = 2.
        distance_BX_expected = 1.
        Y_expected = gs.array([[0, 1]])

        Y_result = MDMEstimator.predict(X)

        self.assertAllClose(Y_result, Y_expected)

        return

# if(__name__=='__main__'):
#     tmp=TestRiemannianMinimumDistanceToMeanClassifier()
#     tmp.test_fit()
#     tmp.test_predict()
