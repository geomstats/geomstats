import random
from random import randint

import pytest

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import Matrices
from geomstats.learning.euclidean import LinearRegression
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDatasetGenerator
from geomstats.test_cases.learning.euclidean import LinearRegressionTestCase

from .data.euclidean import LinearRegressionTestData


def _random_bool():
    return bool(random.getrandbits(1))


@pytest.fixture(
    scope="class",
    params=[
        (
            Euclidean(dim=randint(2, 4)),
            Euclidean(dim=1),
            _random_bool(),
            True,
        ),
        (
            Euclidean(dim=randint(2, 4)),
            Euclidean(dim=1),
            _random_bool(),
            False,
        ),
        (
            Euclidean(dim=randint(2, 4)),
            Euclidean(dim=randint(2, 3)),
            _random_bool(),
            _random_bool(),
        ),
        (
            Euclidean(dim=randint(2, 4)),
            Matrices(randint(2, 4), randint(2, 4)),
            _random_bool(),
            _random_bool(),
        ),
        (
            Matrices(randint(2, 4), randint(2, 4)),
            Euclidean(dim=randint(1, 4)),
            _random_bool(),
            _random_bool(),
        ),
        (
            Matrices(randint(2, 4), randint(2, 4)),
            Matrices(randint(2, 4), randint(2, 4)),
            _random_bool(),
            _random_bool(),
        ),
    ],
)
def estimators(request):
    space, image_space, fit_intercept, positive = request.param
    request.cls.estimator = LinearRegression(
        space, image_space, fit_intercept=fit_intercept, positive=positive
    )

    request.cls.image_space = image_space
    request.cls.data_generator = RandomDatasetGenerator(space, image_space)


@pytest.mark.usefixtures("estimators")
class TestLinearRegression(
    LinearRegressionTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = LinearRegressionTestData()
