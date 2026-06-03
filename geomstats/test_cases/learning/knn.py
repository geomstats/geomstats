import random

import pytest

import geomstats.backend as gs
from geomstats.test_cases.learning._base import BaseEstimatorTestCase


def make_labels(n_samples):
    samples = gs.zeros(n_samples, dtype=int)
    indices = range(n_samples)
    samples[random.sample(indices, n_samples // 2)] = 1

    return samples


class KNearestNeighborsClassifierTestCase(BaseEstimatorTestCase):
    def test_predict(self, X_train, y_train, X_test, y_test):
        self.estimator.fit(X_train, y_train)

        y_pred = self.estimator.predict(X_test)
        self.assertAllEqual(y_pred, y_test)

    def test_predict_proba(self, X_train, y_train, X_test, expected_proba, atol):
        self.estimator.fit(X_train, y_train)

        proba = self.estimator.predict_proba(X_test)
        self.assertAllClose(proba, expected_proba, atol=atol)

    def test_predict_proba_at_label(
        self, X_train, y_train, X_test, expected_at_label, atol
    ):
        self.estimator.fit(X_train, y_train)

        labels = self.estimator.predict(X_test)
        probs = self.estimator.predict_proba(X_test)

        prob_at_label = probs[gs.arange(len(X_test)), labels]
        self.assertAllClose(prob_at_label, expected_at_label, atol=atol)

    def test_score(self, X_train, y_train, X_test, y_test, expected, atol):
        self.estimator.fit(X_train, y_train)

        score = self.estimator.score(X_test, y_test)
        self.assertAllClose(score, expected, atol)


class NeighborClassifierTestCase(KNearestNeighborsClassifierTestCase):
    @pytest.mark.random
    def test_predict_train(self, n_samples):
        """Test exact label recovery when predicting on the training data."""
        X = self.data_generator.random_point(n_points=n_samples)
        y = make_labels(n_samples)

        self.test_predict(X, y, X, y)

    @pytest.mark.random
    def test_predict_proba_train(self, n_samples, atol):
        """Test unit predicted probability for training labels.

        For a 1-neighbor classifier, each training point should select itself as
        nearest neighbor, hence the predicted probability at its label is one.
        """
        X = self.data_generator.random_point(n_points=n_samples)
        y = make_labels(n_samples)

        self.test_predict_proba_at_label(X, y, X, gs.ones(X.shape[0]), atol)

    @pytest.mark.random
    def test_score_is_one(self, n_samples, atol):
        """Test perfect classification score on the training data."""
        X = self.data_generator.random_point(n_points=n_samples)
        y = make_labels(n_samples)

        self.test_score(X, y, X, y, 1.0, atol)
