from tests.conftest import Parametrizer, TestCase, np_backend
from tests.data.aac_data import (
    AACFrechetMeanTestData,
    AACGGPCATestData,
    AACRegressionTestData,
    MaxIterTestData,
)

IS_NOT_NP = not np_backend()


class _MeanEstimatorTest(TestCase):
    def test_fit(self, estimator, X, expected):
        estimate_ = estimator.fit(X).estimate_

        dist = estimator.metric.dist(estimate_, expected)
        self.assertAllClose(dist, 0.0)


class _GeodesicEstimatorTest(TestCase):
    def test_fit(self, estimator, X, atol):
        estimator.fit(X)

        mean = estimator.mean_
        direc = estimator.components_[0]

        new_geo = estimator.metric.total_space_metric.geodesic(
            initial_point=mean, initial_tangent_vec=direc
        )

        dists = estimator.metric.point_to_geodesic_aligner.dist(
            estimator.metric, new_geo, X
        )
        self.assertAllClose(dists, 0.0, atol=atol)


class _RegressionTest(TestCase):
    def test_fit_and_predict(self, estimator, X, y, atol):
        estimator.fit(X, y)

        y_pred = estimator.predict(X)
        dists = estimator.metric.dist(y_pred, y)

        self.assertAllClose(dists, 0.0, atol=atol)


class TestAACFrechetMean(_MeanEstimatorTest, metaclass=Parametrizer):
    skip_all = IS_NOT_NP

    testing_data = AACFrechetMeanTestData()

    def test_fit_id_niter(self, estimator, X):
        estimator.fit(X)

        self.assertEqual(estimator.n_iter_, 2)


class TestAACGGPCA(_GeodesicEstimatorTest, metaclass=Parametrizer):
    skip_all = IS_NOT_NP

    testing_data = AACGGPCATestData()


class TestAACRegression(_RegressionTest, metaclass=Parametrizer):
    skip_all = IS_NOT_NP

    testing_data = AACRegressionTestData()


class TestMaxIter(TestCase, metaclass=Parametrizer):
    skip_all = IS_NOT_NP

    testing_data = MaxIterTestData()

    def test_fit_warn(self, estimator, X, y=None):
        max_iter = estimator.max_iter
        estimator.max_iter = 1

        estimator.fit(X, y=y)
        estimator.max_iter = max_iter
        self.assertEqual(estimator.n_iter_, 1)
