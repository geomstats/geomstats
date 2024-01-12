import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.stratified.graph_space import GraphSpace
from geomstats.learning.aac import _AACGGPCA, AAC, _AACFrechetMean, _AACRegression
from geomstats.test.parametrizers import DataBasedParametrizer, Parametrizer
from geomstats.test.test_case import TestCase
from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
    MeanEstimatorMixinsTestCase,
)
from geomstats.vectorization import repeat_point

from .data.aac import (
    AACFrechetMeanTestData,
    AACGGPCATestData,
    AACRegressionTestData,
    AACTestData,
)


@pytest.mark.smoke
class TestAAC(TestCase, metaclass=Parametrizer):
    total_space = GraphSpace(2, equip=True)
    total_space.equip_with_group_action()
    total_space.equip_with_quotient_structure()

    testing_data = AACTestData()

    def test_init(self, estimate, expected_type):
        estimator = AAC(self.total_space, estimate=estimate)
        self.assertTrue(type(estimator) is expected_type)


class TestAACFrechetMean(
    MeanEstimatorMixinsTestCase, BaseEstimatorTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(3, 4)
    _space = GraphSpace(_n)
    _space.equip_with_group_action()
    _space.equip_with_quotient_structure()
    _space.aligner.set_aligner_algorithm("exhaustive")

    estimator = _AACFrechetMean(_space, init_point=gs.zeros((_n, _n)))

    testing_data = AACFrechetMeanTestData()


class TestAACGGPCA(BaseEstimatorTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(3, 4)
    _space = GraphSpace(_n)
    _space.equip_with_group_action()
    _space.equip_with_quotient_structure()
    _space.aligner.set_aligner_algorithm("exhaustive")

    estimator = _AACGGPCA(_space, init_point=gs.zeros((_n, _n)))

    testing_data = AACGGPCATestData()

    @pytest.mark.random
    def test_fit_geodesic_points(self, n_samples, atol):
        total_space = self.estimator.space

        initial_point, end_point = self.data_generator.random_point(2)
        geod_func = total_space.quotient.metric.geodesic(
            initial_point, end_point=end_point
        )
        s = gs.linspace(0.0, 1.0, n_samples)
        X = geod_func(s)

        self.estimator.fit(X)

        mean = self.estimator.mean_
        direc = self.estimator.components_[0]

        new_geo = total_space.metric.geodesic(
            initial_point=mean, initial_tangent_vec=direc
        )

        dists = total_space.aligner.point_to_geodesic_aligner.dist(
            total_space, new_geo, X
        )
        self.assertAllClose(dists, gs.zeros_like(dists), atol=atol)


class TestAACRegression(BaseEstimatorTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(3, 4)
    _space = GraphSpace(_n)
    _space.equip_with_group_action()
    _space.equip_with_quotient_structure()
    _space.aligner.set_aligner_algorithm("exhaustive")

    estimator = _AACRegression(_space, init_point=gs.zeros((_n, _n)))
    testing_data = AACRegressionTestData()

    @pytest.mark.random
    def test_fit_and_predict_constant(self, n_samples, atol):
        y = repeat_point(self.data_generator.random_point(), n_reps=n_samples)
        X = gs.expand_dims(gs.linspace(0.0, 1.0, num=n_samples), axis=-1)

        self.estimator.fit(X, y)
        y_pred = self.estimator.predict(X)

        total_space = self.estimator.space
        dists = total_space.metric.dist(y_pred, y)

        self.assertAllClose(dists, gs.zeros_like(dists), atol=atol)
