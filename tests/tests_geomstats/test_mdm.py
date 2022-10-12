"""Methods for testing the MDM classifier module."""
import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.spd_matrices import (
    SPDAffineMetric,
    SPDEuclideanMetric,
    SPDLogEuclideanMetric,
)
from geomstats.learning.mdm import RiemannianMinimumDistanceToMean

EULER = gs.exp(1.0)
METRICS = (SPDAffineMetric, SPDLogEuclideanMetric, SPDEuclideanMetric)


class TestRiemannianMinimumDistanceToMeanClassifier(tests.conftest.TestCase):
    """Test of MDM classifier for different metrics."""

    def test_fit(self):
        """Test the fit method."""
        X_train_a = gs.array([[[EULER**2, 0], [0, 1]], [[1, 0], [0, 1]]])
        X_train_b = gs.array([[[EULER**8, 0], [0, 1]], [[1, 0], [0, 1]]])
        X_train = gs.concatenate([X_train_a, X_train_b])
        y_train = gs.array([0, 0, 1, 1])

        for metric in METRICS:
            MDM = RiemannianMinimumDistanceToMean(metric(n=2))
            MDM.fit(X_train, y_train)
            bary_a_fit = MDM.mean_estimates_[0]
            bary_b_fit = MDM.mean_estimates_[1]

            if metric in [SPDAffineMetric, SPDLogEuclideanMetric]:
                bary_a_expected = gs.array([[EULER, 0], [0, 1]])
                bary_b_expected = gs.array([[EULER**4, 0], [0, 1]])
            elif metric in [SPDEuclideanMetric]:
                bary_a_expected = gs.array([[0.5 * EULER**2 + 0.5, 0], [0, 1]])
                bary_b_expected = gs.array([[0.5 * EULER**8 + 0.5, 0], [0, 1]])
            else:
                raise ValueError("Invalid metric: {}".format(metric))

            self.assertAllClose(bary_a_fit, bary_a_expected)
            self.assertAllClose(bary_b_fit, bary_b_expected)

        MDM.fit(X_train, y_train, gs.ones(4))  # with weights
        self.assertAllClose(MDM.mean_estimates_[0], bary_a_expected)
        self.assertAllClose(MDM.mean_estimates_[1], bary_b_expected)

    def test_predict(self):
        """Test the predict method."""
        X_train_a = gs.array([[EULER, 0], [0, 1]])[None, ...]
        X_train_b = gs.array([[EULER**4, 0], [0, 1]])[None, ...]
        X_train = gs.concatenate([X_train_a, X_train_b])
        y_train = gs.array([42, 17])

        X_test = gs.array([[EULER**2, 0], [0, 1]])[None, ...]
        y_expected = gs.array([42])

        for metric in METRICS:
            MDM = RiemannianMinimumDistanceToMean(metric(n=2))
            MDM.fit(X_train, y_train)
            y_test = MDM.predict(X_test)

            self.assertAllClose(y_test, y_expected)

    def test_predict_proba(self):
        """Test the predict_proba method."""
        X_train_a = gs.array([[1.0, 0], [0, 1]])[None, ...]
        X_train_b = gs.array([[EULER**10, 0], [0, 1]])[None, ...]
        X_train = gs.concatenate([X_train_a, X_train_b])
        y_train = gs.array([1, 2])

        X_test = gs.array([[[1.0, 0], [0, 1]], [[EULER**5, 0], [0, 1]]])

        for metric in METRICS:
            MDM = RiemannianMinimumDistanceToMean(metric(n=2))
            MDM.fit(X_train, y_train)
            proba_test = MDM.predict_proba(X_test)

            if metric in [SPDAffineMetric, SPDLogEuclideanMetric]:
                proba_expected = gs.array([[1.0, 0.0], [0.5, 0.5]])
            elif metric in [SPDEuclideanMetric]:
                proba_expected = gs.array([[1.0, 0.0], [1.0, 0.0]])
            else:
                raise ValueError("Invalid metric: {}".format(metric))

            self.assertAllClose(proba_test, proba_expected)

    def test_score(self):
        """Test the score method."""
        X_train_a = gs.array([[EULER, 0], [0, 1]])[None, ...]
        X_train_b = gs.array([[EULER**4, 0], [0, 1]])[None, ...]
        X_train = gs.concatenate([X_train_a, X_train_b])
        y_train = gs.array([-1, 1])

        X_test = gs.array([[[EULER**3, 0], [0, 1]], [[EULER**2, 0], [0, 1]]])

        for metric in METRICS:
            MDM = RiemannianMinimumDistanceToMean(metric(n=2))
            MDM.fit(X_train, y_train)

            if metric in [SPDAffineMetric, SPDLogEuclideanMetric]:
                y_expected = gs.array([1, -1])
            elif metric in [SPDEuclideanMetric]:
                y_expected = gs.array([-1, -1])
            else:
                raise ValueError("Invalid metric: {}".format(metric))

            accuracy = MDM.score(X_test, y_expected)
            self.assertAllClose(accuracy, 1.0)
