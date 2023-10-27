from geomstats.test_cases.learning._base import BaseEstimatorTestCase


class KNearestNeighborsClassifierTestCase(BaseEstimatorTestCase):
    def test_predict(self, X_train, y_train, X_test, y_test):
        self.estimator.fit(X_train, y_train)

        y_pred = self.estimator.predict(X_test)
        self.assertAllEqual(y_pred, y_test)

    def test_predict_proba(self, X_train, y_train, X_test, expected_proba, atol):
        self.estimator.fit(X_train, y_train)

        proba = self.estimator.predict_proba(X_test)
        self.assertAllClose(proba, expected_proba, atol=atol)
