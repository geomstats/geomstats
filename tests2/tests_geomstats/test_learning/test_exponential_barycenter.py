import random

import pytest

from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.exponential_barycenter import (
    ExponentialBarycenter,
    GradientDescent,
)
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
    MeanEstimatorMixinsTestCase,
)
from geomstats.test_cases.learning.exponential_barycenter import EuclideanGroup

from .data.exponential_barycenter import (
    AgainstFrechetMeanTestData,
    AgainstLinearMeanTestData,
    ExponentialBarycenterTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        SpecialEuclidean(n=3, equip=False),
        SpecialOrthogonal(n=3, point_type="vector", equip=False),
        SpecialOrthogonal(n=3, equip=False),
    ],
)
def estimators(request):
    space = request.param
    request.cls.estimator = ExponentialBarycenter(space)


@pytest.mark.usefixtures("estimators")
class TestExponentialBarycenter(
    MeanEstimatorMixinsTestCase, BaseEstimatorTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ExponentialBarycenterTestData()


class TestAgainstFrechetMean(BaseEstimatorTestCase, metaclass=DataBasedParametrizer):
    space = SpecialOrthogonal(n=3, equip=True)
    estimator = ExponentialBarycenter(space)
    other_estimator = FrechetMean(space, method="adaptive")

    testing_data = AgainstFrechetMeanTestData()

    def test_against_frechet_mean(self, n_points, atol):
        X = self.data_generator.random_point(n_points=n_points)

        res = self.estimator.fit(X).estimate_
        res_ = self.other_estimator.fit(X).estimate_
        self.assertAllClose(res, res_, atol=atol)


class TestAgainstLinearMean(BaseEstimatorTestCase, metaclass=DataBasedParametrizer):
    space = EuclideanGroup(dim=random.randint(2, 5), equip=False)
    estimator = ExponentialBarycenter(space)

    testing_data = AgainstLinearMeanTestData()

    def test_against_linear_mean(self, n_points, atol):
        X = self.data_generator.random_point(n_points=n_points)

        optimizer = GradientDescent()
        res = optimizer.minimize(self.estimator.space, X)

        expected = self.estimator.fit(X).estimate_

        self.assertAllClose(res, expected, atol=atol)
