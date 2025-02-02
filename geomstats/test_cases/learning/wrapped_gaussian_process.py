import pytest

import geomstats.backend as gs
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.learning._base import BaseEstimatorTestCase


class WrappedGaussianProcessRandomDataGenerator(RandomDataGenerator):
    def __init__(self, space, prior):
        super().__init__(space)
        self.prior = prior

        self._oscillation = (1.0 / 20.0) * gs.array([-0.5, 0.0, 1.0])

    def random_X(self, n_samples):
        X = gs.linspace(0.0, 1.5 * gs.pi, n_samples)
        return gs.reshape((X - gs.mean(X)), (-1, 1))

    def random_dataset(self, n_samples):
        X = self.random_X(n_samples)

        y = self.prior(X)

        o = self.space.to_tangent(self._oscillation, base_point=y)
        s = X * gs.sin(5.0 * gs.pi * X)

        new_y = self.space.metric.exp(s * o, base_point=y)

        return X, new_y


class WrappedGaussianProcessTestCase(BaseEstimatorTestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = WrappedGaussianProcessRandomDataGenerator(
                self.estimator.space, self.estimator.prior
            )

    @pytest.mark.random
    def test_score_at_train_is_one(self, n_samples, atol):
        X, y = self.data_generator.random_dataset(n_samples)

        self.estimator.fit(X, y)

        res = self.estimator.score(X, y)
        self.assertAllClose(res, 1.0, atol=atol)

    @pytest.mark.random
    def test_predict_at_train_belongs(self, n_samples, atol):
        X, y = self.data_generator.random_dataset(n_samples)

        self.estimator.fit(X, y)

        y_ = self.estimator.predict(X)

        expected = gs.ones(n_samples, dtype=bool)
        res = self.estimator.space.belongs(y_, atol=atol)
        self.assertAllEqual(expected, res)

    @pytest.mark.random
    def test_predict_at_train_zero_std(self, n_samples, atol):
        X, y = self.data_generator.random_dataset(n_samples)

        self.estimator.fit(X, y)

        _, std = self.estimator.predict(X, return_tangent_std=True)

        expected = gs.zeros_like(std)
        self.assertAllClose(std, expected, atol=atol)

    @pytest.mark.random
    def test_sample_y_at_train_belongs(self, n_samples, atol):
        X, y = self.data_generator.random_dataset(n_samples)

        self.estimator.fit(X, y)
        y_ = self.estimator.sample_y(X)
        y_ = gs.reshape(gs.transpose(y_, [0, 2, 1]), (-1, y_.shape[1]))

        res = self.estimator.space.belongs(y_, atol=atol)
        expected = gs.ones(n_samples, dtype=bool)
        self.assertAllEqual(res, expected)
