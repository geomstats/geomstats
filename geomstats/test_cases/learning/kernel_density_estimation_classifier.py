from geomstats.test.test_case import TestCase


class KernelDensityEstimationClassifierTestCase(TestCase):
    def test_predict(self, estimator, X_train, y_train, X_test, y_test):
        estimator.fit(X_train, y_train)

        y_pred = estimator.predict(X_test)
        self.assertAllEqual(y_pred, y_test)

    def test_predict_proba(self, estimator, X_train, y_train, X_test, expected, atol):
        estimator.fit(X_train, y_train)

        proba = estimator.predict_proba(X_test)
        self.assertAllClose(proba, expected, atol=atol)
