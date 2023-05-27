import random

import pytest

from geomstats.information_geometry.normal import (
    CenteredNormalDistributions,
    CenteredNormalMetric,
    DiagonalNormalDistributions,
    DiagonalNormalMetric,
    GeneralNormalDistributions,
    UnivariateNormalDistributions,
    UnivariateNormalMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.information_geometry.normal import (
    CenteredNormalDistributionsTestCase,
    CenteredNormalMetricTestCase,
    DiagonalNormalDistributionsTestCase,
    DiagonalNormalMetricTestCase,
    GeneralNormalDistributionsTestCase,
    GeneralNormalMetricTestCase,
    UnivariateNormalDistributionsTestCase,
    UnivariateNormalMetricTestCase,
)
from tests2.tests_geomstats.test_information_geometry.data.normal import (
    CenteredNormalDistributionsTestData,
    CenteredNormalMetricTestData,
    DiagonalNormalDistributionsTestData,
    DiagonalNormalMetricTestData,
    GeneralNormalDistributionsTestData,
    GeneralNormalMetricTestData,
    UnivariateNormalDistributionsTestData,
    UnivariateNormalMetricTestData,
)


class TestUnivariateNormalDistributions(
    UnivariateNormalDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = UnivariateNormalDistributions(equip=False)
    testing_data = UnivariateNormalDistributionsTestData()


class TestUnivariateNormalMetric(
    UnivariateNormalMetricTestCase, metaclass=DataBasedParametrizer
):
    space = UnivariateNormalDistributions(equip=False)
    space.equip_with_metric(UnivariateNormalMetric)

    data_generator = RandomDataGenerator(space, amplitude=5.0)
    data_generator_embedding = RandomDataGenerator(space.metric.embedding_space)

    testing_data = UnivariateNormalMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def centered_spaces(request):
    request.cls.space = CenteredNormalDistributions(
        sample_dim=request.param, equip=False
    )


@pytest.mark.usefixtures("centered_spaces")
class TestCenteredNormalDistributions(
    CenteredNormalDistributionsTestCase, metaclass=DataBasedParametrizer
):
    testing_data = CenteredNormalDistributionsTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_centered_spaces(request):
    space = request.cls.space = CenteredNormalDistributions(
        sample_dim=request.param, equip=False
    )
    space.equip_with_metric(CenteredNormalMetric)


@pytest.mark.usefixtures("equipped_centered_spaces")
class TestCenteredNormalMetric(
    CenteredNormalMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = CenteredNormalMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def diagonal_spaces(request):
    request.cls.space = DiagonalNormalDistributions(
        sample_dim=request.param, equip=False
    )


@pytest.mark.usefixtures("diagonal_spaces")
class TestDiagonalNormalDistributions(
    DiagonalNormalDistributionsTestCase, metaclass=DataBasedParametrizer
):
    testing_data = DiagonalNormalDistributionsTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_diagonal_spaces(request):
    space = request.cls.space = DiagonalNormalDistributions(
        sample_dim=request.param, equip=False
    )
    space.equip_with_metric(DiagonalNormalMetric)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=10.0)


@pytest.mark.usefixtures("equipped_diagonal_spaces")
class TestDiagonalNormalMetric(
    DiagonalNormalMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = DiagonalNormalMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def general_spaces(request):
    request.cls.space = GeneralNormalDistributions(
        sample_dim=request.param, equip=False
    )


@pytest.mark.usefixtures("general_spaces")
class TestGeneralNormalDistributions(
    GeneralNormalDistributionsTestCase, metaclass=DataBasedParametrizer
):
    testing_data = GeneralNormalDistributionsTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_general_spaces(request):
    request.cls.space = GeneralNormalDistributions(sample_dim=request.param, equip=True)


@pytest.mark.skip
@pytest.mark.usefixtures("equipped_general_spaces")
class TestGeneralNormalMetric(
    GeneralNormalMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = GeneralNormalMetricTestData()

    # TODO: come back here
