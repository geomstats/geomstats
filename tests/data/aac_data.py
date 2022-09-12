import geomstats.backend as gs
from geomstats.geometry.stratified.graph_space import (
    ExhaustiveAligner,
    GraphSpace,
    GraphSpaceMetric,
    _GeodesicToPointAligner,
)
from geomstats.learning.aac import _AACGGPCA, _AACFrechetMean, _AACRegression
from tests.data_generation import TestData


class _TrivialMeanEstimatorTestData(TestData):
    def __init__(self):
        self.metrics = None
        self.estimators = None
        self.n_reps = 2
        self._setup()

    def fit_test_data(self):
        smoke_data = []
        for metric, estimator in zip(self.metrics, self.estimators):
            point = metric.space.random_point()
            points = gs.repeat(gs.expand_dims(point, 0), self.n_reps, axis=0)

            datum = dict(estimator=estimator, X=points, expected=point)
            smoke_data.append(datum)

        return self.generate_tests(smoke_data)


class _TrivialGeodesicEstimatorTestData(TestData):
    def __init__(self):
        self.metrics = []
        self.estimators = []
        self.n_points = 3
        self._setup()

    def fit_test_data(self):
        smoke_data = []
        for metric, estimator in zip(self.metrics, self.estimators):
            init_point, end_point = metric.space.random_point(2)
            geo = metric.geodesic(init_point, end_point)

            s = gs.linspace(0.0, 1.0, self.n_points)
            points = geo(s)

            datum = dict(estimator=estimator, X=points)
            smoke_data.append(datum)

        return self.generate_tests(smoke_data)


class _TrivialRegressionTestData(TestData):
    def __init__(self):
        self.n_samples = 3
        self.input_dim = []
        self.metrics = []
        self.estimators = []
        self._setup()

    def fit_and_predict_test_data(self):
        smoke_data = []

        for p, metric, estimator in zip(self.input_dim, self.metrics, self.estimators):
            y = gs.repeat(
                gs.expand_dims(metric.space.random_point(), 0),
                self.n_samples,
                axis=0,
            )
            X = gs.repeat(
                gs.reshape(gs.linspace(0.0, 1.0, num=self.n_samples), (-1, 1)),
                p,
                axis=1,
            )
            datum = dict(estimator=estimator, X=X, y=y)
            smoke_data.append(datum)

        return self.generate_tests(smoke_data)


class AACFrechetMeanTestData(_TrivialMeanEstimatorTestData):
    def _setup(self):
        space_2 = GraphSpace(2)

        metric = GraphSpaceMetric(space_2)
        metric.set_aligner(ExhaustiveAligner())

        self.metrics = [metric] * 2

        self.estimators = [_AACFrechetMean(metric) for metric in self.metrics]
        self.estimators[0].init_point = gs.zeros((2, 2))

    def fit_id_niter_test_data(self):
        space_3 = GraphSpace(3)
        metric = GraphSpaceMetric(space_3)

        smoke_data = [
            dict(estimator=_AACFrechetMean(metric), X=space_3.random_point(4)),
        ]

        return self.generate_tests(smoke_data)


class AACGGPCATestData(_TrivialGeodesicEstimatorTestData):
    tolerances = {
        "fit": {"atol": 1e-8},
    }

    def _setup(self):
        space_2 = GraphSpace(2)

        metric = GraphSpaceMetric(space_2)
        metric.set_aligner(ExhaustiveAligner())

        aligner = _GeodesicToPointAligner()
        metric.set_point_to_geodesic_aligner(aligner)

        self.metrics = [metric] * 2
        self.estimators = [_AACGGPCA(metric) for metric in self.metrics]

        self.estimators[-1].init_point = gs.zeros((2, 2))


class AACRegressionTestData(_TrivialRegressionTestData):
    def _setup(self):

        space_2 = GraphSpace(2)

        metric = GraphSpaceMetric(space_2)
        metric.set_aligner(ExhaustiveAligner())

        metric.set_point_to_geodesic_aligner(
            "default", s_min=-1.0, s_max=1.0, n_points=10
        )

        self.metrics = [metric] * 3
        self.estimators = [_AACRegression(metric) for metric in self.metrics]

        self.estimators[1].init_point = gs.zeros((2, 2))
        self.input_dim = [1, 1, 2]


class MaxIterTestData(TestData):
    def fit_warn_test_data(self):
        n_samples = 3
        space_2 = GraphSpace(2)
        metric = GraphSpaceMetric(space_2)
        metric.set_point_to_geodesic_aligner("default")

        X = space_2.random_point(n_samples)
        inputs = [X] * 2 + [gs.expand_dims(gs.linspace(0, 1, num=n_samples), 1)]
        outputs = [None] * 2 + [X]

        estimators = [
            _AACFrechetMean(metric),
            _AACGGPCA(metric),
            _AACRegression(metric),
        ]

        smoke_data = []
        for estimator, X, y in zip(estimators, inputs, outputs):
            smoke_data.append(dict(estimator=estimator, X=X, y=y))

        return self.generate_tests(smoke_data)
