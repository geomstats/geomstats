import pytest

from geomstats.test_cases.learning._base import BaseEstimatorTestCase


class LinearRegressionTestCase(BaseEstimatorTestCase):
    @pytest.mark.random
    def test_runs(self, n_samples):
        X = self.data_generator.random_point(n_points=n_samples)
        y = self.data_generator.random_image_point(n_points=n_samples).squeeze()

        self.estimator.fit(X, y)

        X_pred = self.estimator.predict(X)
        if X_pred.ndim > 1:
            self.assertEqual(X_pred.shape, (n_samples,) + self.image_space.shape)
        else:
            self.assertEqual(X_pred.shape, (n_samples,))

        self.estimator.score(X, y)
