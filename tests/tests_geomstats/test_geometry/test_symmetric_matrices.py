import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.symmetric_matrices import (
    ConstantValueRowSumsDiffeo,
    HollowMatricesPermutationInvariantMetric,
    NullRowSumsPermutationInvariantMetric,
    NullRowSumsSymmetricMatrices,
    SymmetricHollowMatrices,
    SymmetricMatrices,
)
from geomstats.test_cases.geometry.base import (
    LevelSetTestCase,
    MatrixVectorSpaceTestCase,
)
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.euclidean import EuclideanMetricTestCase
from geomstats.test_cases.geometry.matrices import MatricesMetricTestCase

from .data.base import MatrixVectorSpaceTestData
from .data.diffeo import DiffeoTestData
from .data.euclidean import EuclideanMetricTestData
from .data.matrices import MatricesMetricTestData
from .data.symmetric_matrices import (
    NullRowSumsSymmetricMatricesTestData,
    SymmetricHollowMatricesTestData,
    SymmetricMatrices1TestData,
    SymmetricMatrices2TestData,
    SymmetricMatrices3TestData,
)


class TestSymmetricMatrices(
    MatrixVectorSpaceTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n = random.randint(2, 5)
    space = SymmetricMatrices(n=_n, equip=False)
    testing_data = MatrixVectorSpaceTestData(space)


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
class TestMatricesMetric(
    MatricesMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(2, 5)
    space = SymmetricMatrices(n=_n)

    testing_data = MatricesMetricTestData(space)


class TestSymmetricHollowMatrices(
    LevelSetTestCase,
    MatrixVectorSpaceTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n = random.randint(2, 6)
    space = SymmetricHollowMatrices(n=_n, equip=False)

    testing_data = SymmetricHollowMatricesTestData(space)


@pytest.fixture(
    scope="class",
    params=[2, 3, random.randint(4, 6)],
)
def equipped_hollow_matrices(request):
    n = request.param
    space = request.cls.space = SymmetricHollowMatrices(
        n, equip=False
    ).equip_with_metric(HollowMatricesPermutationInvariantMetric)

    request.cls.testing_data.space = space


@pytest.mark.usefixtures("equipped_hollow_matrices")
class TestHollowMatricesPermutationInvariantMetric(
    EuclideanMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = EuclideanMetricTestData(
        excluded_methods=(
            "christoffels_are_zeros",
            "inner_product_derivative_matrix_is_zeros",
        ),
    )


class TestNullRowSumSymmetricMatrices(
    LevelSetTestCase,
    MatrixVectorSpaceTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n = random.randint(3, 6)
    space = NullRowSumsSymmetricMatrices(n=_n, equip=False)

    testing_data = NullRowSumsSymmetricMatricesTestData(space)


class TestNullRowSumsDiffeo(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    _n = random.randint(3, 6)
    domain_space = NullRowSumsSymmetricMatrices(n=_n, equip=False)
    image_space = SymmetricMatrices(n=_n - 1, equip=False)
    diffeo = ConstantValueRowSumsDiffeo()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


@pytest.fixture(
    scope="class",
    params=[2, 3, random.randint(4, 6)],
)
def equipped_null_row_sums_matrices(request):
    n = request.param
    space = request.cls.space = NullRowSumsSymmetricMatrices(
        n, equip=False
    ).equip_with_metric(NullRowSumsPermutationInvariantMetric)

    request.cls.testing_data.space = space


@pytest.mark.usefixtures("equipped_null_row_sums_matrices")
class TestNullRowSumsPermutationInvariantMetric(
    EuclideanMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = EuclideanMetricTestData(
        excluded_methods=(
            "christoffels_are_zeros",
            "inner_product_derivative_matrix_is_zeros",
        ),
    )
