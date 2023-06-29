import pytest

import geomstats.backend as gs
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.learning._base import BaseEstimatorTestCase


class Model:
    # TODO: bring generation with noise
    def __init__(self, space, intercept, coef):
        self.space = space
        self.intercept = intercept
        self.coef = coef

    @property
    def param(self):
        return gs.vstack([self.intercept, self.coef])

    def generate(self, n_samples):
        time = gs.random.rand(n_samples)
        time = time - gs.mean(time)

        y = self.space.metric.exp(
            gs.einsum("n,...->n...", time, self.coef), self.intercept
        )
        return time, y


class GeodesicRegressionRandomDataGenerator(RandomDataGenerator):
    def random_model(self):
        intercept_true = self.random_point()
        coef_true = self.random_tangent_vec(intercept_true)

        return Model(self.space, intercept_true, coef_true)

    def random_X(self, n_samples):
        time = gs.random.rand(n_samples)
        time = time - gs.mean(time)
        return time

    def random_dataset(self, n_samples):
        # TODO: distinction between noisy and noiseless?
        model = self.random_model()
        X, y = model.generate(n_samples)
        return X, y


class GeodesicRegressionTestCase(BaseEstimatorTestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = GeodesicRegressionRandomDataGenerator(
                self.estimator.space
            )

    @pytest.mark.random
    def test_loss(self, n_samples, atol):
        model = self.data_generator.random_model()
        X, y = model.generate(n_samples)

        loss = self.estimator._loss(X, y, model.param)
        self.assertAllClose(loss, 0.0, atol=atol)

    @pytest.mark.random
    def test_param_belongs_and_is_tangent(self, n_samples, atol):
        space = self.estimator.space
        X, y = self.data_generator.random_dataset(n_samples)

        self.estimator.fit(X, y)

        intercept = self.estimator.intercept_
        coef = self.estimator.coef_

        belongs = space.belongs(intercept, atol=atol)
        self.assertTrue(belongs)

        is_tangent = space.is_tangent(coef, intercept, atol=atol)
        self.assertTrue(is_tangent)

    @pytest.mark.random
    def test_predict_and_score(self, n_samples, atol):
        X, y = self.data_generator.random_dataset(n_samples)  # assumes noiseless

        self.estimator.fit(X, y)

        score = self.estimator.score(X, y)
        self.assertTrue(gs.isclose(score, 1.0, atol=atol), msg=f"score: {score}")
