import random

import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import Matrices
from geomstats.test_cases.numerics.finite_differences import (
    CenteredDifferenceTestCase,
    ForwardDifferenceTestCase,
    SecondCenteredDifferenceTestCase,
)

from .data.finite_differences import (
    CenteredDifferenceTestData,
    ForwardDifferenceTestData,
    SecondCenteredDifferenceTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        Euclidean(dim=random.randint(2, 5), equip=False),
        Matrices(m=random.randint(2, 5), n=random.randint(2, 5)),
    ],
)
def spaces(request):
    space = request.cls.space = request.param
    request.cls.testing_data.space = space


@pytest.mark.usefixtures("spaces")
class TestForwardDifference(
    ForwardDifferenceTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = ForwardDifferenceTestData()


@pytest.mark.usefixtures("spaces")
class TestCenteredDifference(
    CenteredDifferenceTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = CenteredDifferenceTestData()


@pytest.mark.usefixtures("spaces")
class TestSecondCenteredDifference(
    SecondCenteredDifferenceTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = SecondCenteredDifferenceTestData()
