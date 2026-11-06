import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.full_rank_matrices import FullRankMatrices
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.matrices import MatricesMetricTestCase

from .data.base import VectorSpaceOpenSetTestData
from .data.full_rank_matrices import FullRankMatrices32TestData
from .data.matrices import MatricesMetricTestData


@pytest.mark.smoke
class TestFullRankMatrices32(
    VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer
):
    space = FullRankMatrices(3, 2, equip=False)
    testing_data = FullRankMatrices32TestData()


class TestFullRankMatrices(
    VectorSpaceOpenSetTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = FullRankMatrices(random.randint(3, 5), random.randint(3, 5), equip=False)
    testing_data = VectorSpaceOpenSetTestData(space)


@pytest.mark.redundant
class TestMatricesMetric(
    MatricesMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = FullRankMatrices(random.randint(3, 5), random.randint(3, 5))
    testing_data = MatricesMetricTestData(space)
