import random

import pytest

from geomstats.geometry.invariant_metric import (
    _InvariantMetricMatrix,
    _InvariantMetricVector,
)
from geomstats.geometry.special_orthogonal import (
    SpecialOrthogonal,
    _SpecialOrthogonalMatrices,
)
from geomstats.test.geometry.invariant_metric import (
    InvariantMetricMatrixTestCase,
    InvariantMetricVectorTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.invariant_metric_data import (
    InvariantMetricMatrixTestData,
    InvariantMetricVectorTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        _SpecialOrthogonalMatrices(2, equip=False),
        _SpecialOrthogonalMatrices(random.randint(3, 5), equip=False),
    ],
)
def equipped_matrix_groups(request):
    # TODO: do for left and right
    space = request.cls.space = request.param
    space.equip_with_metric(_InvariantMetricMatrix)


@pytest.mark.usefixtures("equipped_matrix_groups")
class TestInvariantMetricMatrix(
    InvariantMetricMatrixTestCase, metaclass=DataBasedParametrizer
):
    testing_data = InvariantMetricMatrixTestData()


@pytest.fixture(
    scope="class",
    params=[
        SpecialOrthogonal(2, point_type="vector", equip=False),
        SpecialOrthogonal(3, point_type="vector", equip=False),
    ],
)
def equipped_vector_spaces(request):
    # TODO: do for left and right
    space = request.cls.space = request.param
    space.equip_with_metric(_InvariantMetricVector)


@pytest.mark.usefixtures("equipped_vector_spaces")
class TestInvariantMetricVector(
    InvariantMetricVectorTestCase, metaclass=DataBasedParametrizer
):
    testing_data = InvariantMetricVectorTestData()
