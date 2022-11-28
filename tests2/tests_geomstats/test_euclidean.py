import random

import pytest

from geomstats.geometry.euclidean import Euclidean
from geomstats.test.geometry.euclidean import EuclideanTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.euclidean_data import EuclideanTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = Euclidean(dim=request.param)


@pytest.mark.usefixtures("spaces")
class TestEuclidean(EuclideanTestCase, metaclass=DataBasedParametrizer):
    testing_data = EuclideanTestData()
