import random

import pytest

from geomstats.geometry.symmetric_matrices import (
    HollowMatricesPermutationInvariantMetric,
    NullRowSumDiffeo,
    NullRowSumSymmetricMatrices,
    SymmetricHollowMatrices,
    SymmetricMatrices,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import (
    LevelSetTestCase,
    MatrixVectorSpaceTestCase,
)
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.euclidean import EuclideanMetricTestCase
from geomstats.test_cases.geometry.matrices import MatricesMetricTestCase

from .data.diffeo import DiffeoTestData
from .data.matrices import MatricesMetricTestData
from .data.symmetric_matrices import (
    HollowMatricesPermutationInvariantMetricTestData,
    NullRowSumSymmetricMatricesTestData,
    SymmetricHollowMatricesTestData,
    SymmetricMatrices1TestData,
    SymmetricMatrices2TestData,
    SymmetricMatrices3TestData,
    SymmetricMatricesTestData,
)


class TestSymmetricMatrices(
    MatrixVectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    _n = random.randint(2, 5)
    space = SymmetricMatrices(n=_n, equip=False)
    testing_data = SymmetricMatricesTestData()


@pytest.mark.smoke
class TestSymmetricMatrices1(
    MatrixVectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    space = SymmetricMatrices(n=1, equip=False)
    testing_data = SymmetricMatrices1TestData()


@pytest.mark.smoke
class TestSymmetricMatrices2(
    MatrixVectorSpaceTestCase, metaclass=DataBasedParametrizer
):
    space = SymmetricMatrices(n=2, equip=False)
    testing_data = SymmetricMatrices2TestData()


@pytest.mark.smoke
class TestSymmetricMatrices3(
    MatrixVectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    space = SymmetricMatrices(n=3, equip=False)
    testing_data = SymmetricMatrices3TestData()


@pytest.mark.redundant
class TestMatricesMetric(MatricesMetricTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)
    space = SymmetricMatrices(n=_n)

    testing_data = MatricesMetricTestData()


class TestSymmetricHollowMatrices(
    LevelSetTestCase,
    MatrixVectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    _n = random.randint(2, 6)
    space = SymmetricHollowMatrices(n=_n, equip=False)

    testing_data = SymmetricHollowMatricesTestData()


@pytest.fixture(
    scope="class",
    params=[
        (2, (0.0, 0.0, 1.0)),
        (3, (0.0, 1.0, 1.0)),
        (random.randint(4, 6), (1.0, 1.0, 1.0)),
    ],
)
def equipped_hollow_matrices(request):
    n, (alpha, beta, gamma) = request.param
    request.cls.space = SymmetricHollowMatrices(n, equip=False).equip_with_metric(
        HollowMatricesPermutationInvariantMetric, alpha=alpha, beta=beta, gamma=gamma
    )


@pytest.mark.usefixtures("equipped_hollow_matrices")
class TestHollowMatricesPermutationInvariantMetric(
    EuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HollowMatricesPermutationInvariantMetricTestData()


class TestNullRowSumSymmetricMatrices(
    LevelSetTestCase,
    MatrixVectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    _n = random.randint(2, 6)
    space = NullRowSumSymmetricMatrices(n=_n, equip=False)

    testing_data = NullRowSumSymmetricMatricesTestData()


class TestNullRowSumDiffeo(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)
    space = NullRowSumSymmetricMatrices(n=_n, equip=False)
    image_space = SymmetricMatrices(n=_n - 1, equip=False)
    diffeo = NullRowSumDiffeo()
    testing_data = DiffeoTestData()
