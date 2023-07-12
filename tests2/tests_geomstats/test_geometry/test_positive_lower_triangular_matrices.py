import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.positive_lower_triangular_matrices import (
    CholeskyMetric,
    InvariantPositiveLowerTriangularMatricesMetric,
    PositiveLowerTriangularMatrices,
)
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.positive_lower_triangular_matrices import (
    CholeskyMetricTestCase,
    InvariantPositiveLowerTriangularMatricesMetricTestCase,
    PositiveLowerTriangularMatricesTestCase,
)
from tests2.tests_geomstats.test_geometry.data.positive_lower_triangular_matrices import (
    CholeskyMetricTestData,
    InvariantPositiveLowerTriangularMatricesMetricTestData,
)

from .data.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatricesTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 3),
        random.randint(4, 5),
    ],
)
def spaces(request):
    request.cls.space = PositiveLowerTriangularMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestPositiveLowerTriangularMatrices(
    PositiveLowerTriangularMatricesTestCase, metaclass=DataBasedParametrizer
):
    testing_data = PositiveLowerTriangularMatricesTestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 3),
        random.randint(4, 5),
    ],
)
def equipped_spaces(request):
    space = request.cls.space = PositiveLowerTriangularMatrices(
        n=request.param, equip=False
    )
    space.equip_with_metric(CholeskyMetric)


@pytest.mark.usefixtures("equipped_spaces")
class TestCholeskyMetric(CholeskyMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = CholeskyMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        (3, True, True),
        (3, True, False),
    ],
)
def spaces_with_invariant(request):
    n, left, identity = request.param

    space = request.cls.space = PositiveLowerTriangularMatrices(n, equip=False)

    if identity:
        metric_mat_at_identity = gs.eye(space.dim)
    else:
        metric_mat_at_identity = SPDMatrices(n=space.dim, equip=False).random_point()

    space.equip_with_metric(
        InvariantPositiveLowerTriangularMatricesMetric,
        metric_mat_at_identity=metric_mat_at_identity,
        left=left,
    )


@pytest.mark.usefixtures("spaces_with_invariant")
class TestInvariantPositiveLowerTriangularMatricesMetric(
    InvariantPositiveLowerTriangularMatricesMetricTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = InvariantPositiveLowerTriangularMatricesMetricTestData()
