from geomstats.test_cases.learning._base import BaseEstimatorTestCase


class RiemannianMinimumDistanceToMeanTestCase(BaseEstimatorTestCase):
    def test_fit(self, X_train, y_train, expected, atol):
        mean_estimates = self.estimator.fit(X_train, y_train).mean_estimates_
        self.assertAllClose(mean_estimates, expected, atol=atol)

    def test_predict(self, X_train, y_train, X_test, y_test, atol):
        self.estimator.fit(X_train, y_train)

        y_pred = self.estimator.predict(X_test)
        self.assertAllClose(y_pred, y_test, atol=atol)

    def test_predict_proba(self, X_train, y_train, X_test, expected_proba, atol):
        self.estimator.fit(X_train, y_train)

        proba = self.estimator.predict_proba(X_test)
        self.assertAllClose(proba, expected_proba, atol=atol)

    def test_transform(self, X_train, y_train, X_test, expected, atol):
        self.estimator.fit(X_train, y_train)

        dists = self.estimator.transform(X_test)
        self.assertAllClose(dists, expected, atol=atol)

    def test_score(self, X_train, y_train, X_test, y_expected, atol):
        self.estimator.fit(X_train, y_train)

        score = self.estimator.score(X_test, y_expected)
        self.assertAllClose(score, 1.0, atol=atol)
