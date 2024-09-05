import random

import pytest

from geomstats.geometry.lower_triangular_matrices import StrictlyLowerTriangularMatrices
from geomstats.geometry.positive_lower_triangular_matrices import (
    CholeskyMetric,
    InvariantPositiveLowerTriangularMatricesMetric,
    LowerMatrixLog,
    PLTUnitDiagMatrices,
    PositiveLowerTriangularMatrices,
    UnitNormedRowsPLTDiffeo,
    UnitNormedRowsPLTMatrices,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.base import (
    DiffeomorphicManifoldTestCase,
    LevelSetTestCase,
    VectorSpaceOpenSetTestCase,
)
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.invariant_metric import InvariantMetricMatrixTestCase
from geomstats.test_cases.geometry.lie_group import MatrixLieGroupTestCase
from geomstats.test_cases.geometry.positive_lower_triangular_matrices import (
    CholeskyMetricTestCase,
)
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase

from .data.base import DiffeomorphicManifoldTestData
from .data.diffeo import DiffeoTestData
from .data.positive_lower_triangular_matrices import (
    CholeskyMetric2TestData,
    CholeskyMetricTestData,
    InvariantPositiveLowerTriangularMatricesMetricTestData,
    PLTUnitDiagMatricesTestData,
    PositiveLowerTriangularMatrices2TestData,
    PositiveLowerTriangularMatricesTestData,
    UnitNormedRowsPLTMatricesPullbackMetricTestData,
)


class TestPositiveLowerTriangularMatrices(
    MatrixLieGroupTestCase, VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 5)
    space = PositiveLowerTriangularMatrices(n=_n, equip=False)
    testing_data = PositiveLowerTriangularMatricesTestData()


@pytest.mark.smoke
class TestPositiveLowerTriangularMatrices2(
    MatrixLieGroupTestCase, VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer
):
    space = PositiveLowerTriangularMatrices(n=2, equip=False)
    testing_data = PositiveLowerTriangularMatrices2TestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
    ],
)
def equipped_spaces(request):
    request.cls.space = PositiveLowerTriangularMatrices(n=request.param)


@pytest.mark.usefixtures("equipped_spaces")
class TestCholeskyMetric(CholeskyMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = CholeskyMetricTestData()


@pytest.mark.smoke
class TestCholeskyMetric2(CholeskyMetricTestCase, metaclass=DataBasedParametrizer):
    space = PositiveLowerTriangularMatrices(n=2, equip=False)
    space.equip_with_metric(CholeskyMetric)
    testing_data = CholeskyMetric2TestData()


@pytest.mark.slow
@pytest.mark.redundant
class TestInvariantPositiveLowerTriangularMatricesMetric(
    InvariantMetricMatrixTestCase, metaclass=DataBasedParametrizer
):
    space = PositiveLowerTriangularMatrices(n=2, equip=False).equip_with_metric(
        InvariantPositiveLowerTriangularMatricesMetric
    )
    testing_data = InvariantPositiveLowerTriangularMatricesMetricTestData()


class TestUnitNormedRowsPLTDiffeo(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)
    space = UnitNormedRowsPLTMatrices(n=_n, equip=False)
    image_space = space.image_space
    diffeo = UnitNormedRowsPLTDiffeo(_n)
    testing_data = DiffeoTestData()


class TestUnitNormedRowsPLTMatrices(
    DiffeomorphicManifoldTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 5)
    space = UnitNormedRowsPLTMatrices(n=_n, equip=False)
    testing_data = DiffeomorphicManifoldTestData()


class TestUnitNormedRowsPLTMatricesPullbackMetric(
    PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 5)
    space = UnitNormedRowsPLTMatrices(n=_n)
    data_generator = RandomDataGenerator(space, amplitude=5.0)
    testing_data = UnitNormedRowsPLTMatricesPullbackMetricTestData()


class TestPLTUnitDiagMatrices(LevelSetTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)
    space = PLTUnitDiagMatrices(n=_n, equip=False)
    testing_data = PLTUnitDiagMatricesTestData()


class TestLowerMatrixLog(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)

    space = PLTUnitDiagMatrices(n=_n, equip=False)
    image_space = StrictlyLowerTriangularMatrices(n=_n, equip=False)
    diffeo = LowerMatrixLog()
    testing_data = DiffeoTestData()
