import geomstats.backend as gs
from geomstats.learning.aac import AAC
from geomstats.test.parametrizers import Parametrizer
from geomstats.test.test_case import TestCase

from .data.aac import (
    AACFrechetMeanTestData,
    AACGGPCATestData,
    AACRegressionTestData,
    AACTestData,
)


class _TestEstimator(TestCase):
    def test_fit_warn(self, estimator, X, y=None):
        max_iter = estimator.max_iter
        estimator.max_iter = 1

        estimator.fit(X, y=y)
        estimator.max_iter = max_iter
        self.assertEqual(estimator.n_iter_, 1)


class TestAAC(TestCase, metaclass=Parametrizer):
    testing_data = AACTestData()

    def test_init(self, estimate, metric, expected_type):
        estimator = AAC(metric, estimate=estimate)
        self.assertTrue(type(estimator) is expected_type)


class TestAACFrechetMean(_TestEstimator, metaclass=Parametrizer):
    testing_data = AACFrechetMeanTestData()

    def test_fit(self, estimator, X, expected):
        estimate_ = estimator.fit(X, y=None).estimate_

        dist = estimator.metric.dist(estimate_, expected)
        self.assertAllClose(dist, 0.0)

    def test_fit_id_niter(self, estimator, X):
        estimator.fit(X)

        self.assertEqual(estimator.n_iter_, 2)


class TestAACGGPCA(_TestEstimator, metaclass=Parametrizer):
    testing_data = AACGGPCATestData()

    def test_fit(self, estimator, X, atol):
        estimator.fit(X)

        mean = estimator.mean_
        direc = estimator.components_[0]

        new_geo = estimator.metric.total_space_metric.geodesic(
            initial_point=mean, initial_tangent_vec=direc
        )

        dists = estimator.metric.point_to_geodesic_aligner.dist(estimator, new_geo, X)
        self.assertAllClose(dists, gs.zeros_like(dists), atol=atol)


class TestAACRegression(_TestEstimator, metaclass=Parametrizer):
    testing_data = AACRegressionTestData()

    def test_fit_and_predict(self, estimator, X, y, atol):
        estimator.fit(X, y)

        y_pred = estimator.predict(X)
        dists = estimator.metric.dist(y_pred, y)

        self.assertAllClose(dists, gs.zeros_like(dists), atol=atol)
