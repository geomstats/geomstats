import random

import pytest

from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.information_geometry.normal import (
    DiagonalNormalDistributionsRandomVariable,
    MultivariateNormalDistributionsRandomVariable,
    NormalDistributions,
    SharedMeanNormalDistributionsRandomVariable,
    UnivariateNormalDistributionsRandomVariable,
    UnivariateNormalToPoincareHalfSpaceDiffeo,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.poincare_half_space import PoincareHalfSpaceTestCase
from geomstats.test_cases.geometry.product_manifold import ProductManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.geometry.spd_matrices import SPDMatricesTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)
from geomstats.test_cases.information_geometry.normal import (
    UnivariateNormalMetricTestCase,
)

from ..test_geometry.data.diffeo import DiffeoTestData
from ..test_geometry.data.spd_matrices import SPDAffineMetricTestData
from .data.normal import (
    CenteredNormalDistributionsTestData,
    DiagonalNormalDistributionsTestData,
    DiagonalNormalMetricTestData,
    GeneralNormalDistributionsTestData,
    GeneralNormalMetricTestData,
    UnivariateNormalDistributionsTestData,
    UnivariateNormalMetricTestData,
)


class TestUnivariateNormalDistributions(
    InformationManifoldMixinTestCase,
    PoincareHalfSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    space = NormalDistributions(sample_dim=1, equip=False)
    random_variable = UnivariateNormalDistributionsRandomVariable(space)
    testing_data = UnivariateNormalDistributionsTestData()


class TestUnivariateNormalToPoincareHalfSpaceDiffeo(
    DiffeoTestCase, metaclass=DataBasedParametrizer
):
    space = NormalDistributions(sample_dim=1, equip=False)
    image_space = PoincareHalfSpace(dim=2, equip=False)
    diffeo = UnivariateNormalToPoincareHalfSpaceDiffeo()
    testing_data = DiffeoTestData()


class TestUnivariateNormalMetric(
    UnivariateNormalMetricTestCase, metaclass=DataBasedParametrizer
):
    space = NormalDistributions(sample_dim=1)
    data_generator = RandomDataGenerator(space, amplitude=5.0)

    testing_data = UnivariateNormalMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(3, 5),
    ],
)
def centered_spaces(request):
    space = request.cls.space = NormalDistributions(
        sample_dim=request.param, distribution_type="centered", equip=False
    )
    request.cls.random_variable = SharedMeanNormalDistributionsRandomVariable(space)


@pytest.mark.usefixtures("centered_spaces")
class TestCenteredNormalDistributions(
    InformationManifoldMixinTestCase,
    SPDMatricesTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = CenteredNormalDistributionsTestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(3, 5),
    ],
)
def equipped_centered_spaces(request):
    request.cls.space = NormalDistributions(
        sample_dim=request.param, distribution_type="centered"
    )


@pytest.mark.redundant
@pytest.mark.usefixtures("equipped_centered_spaces")
class TestCenteredNormalMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDAffineMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def diagonal_spaces(request):
    space = request.cls.space = NormalDistributions(
        sample_dim=request.param, distribution_type="diagonal", equip=False
    )
    request.cls.random_variable = DiagonalNormalDistributionsRandomVariable(space)


@pytest.mark.usefixtures("diagonal_spaces")
class TestDiagonalNormalDistributions(
    InformationManifoldMixinTestCase,
    VectorSpaceOpenSetTestCase,
    metaclass=DataBasedParametrizer,
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
    space = request.cls.space = NormalDistributions(
        sample_dim=request.param, distribution_type="diagonal"
    )
    request.cls.data_generator = RandomDataGenerator(space, amplitude=10.0)


@pytest.mark.usefixtures("equipped_diagonal_spaces")
class TestDiagonalNormalMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = DiagonalNormalMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(3, 5),
    ],
)
def general_spaces(request):
    space = request.cls.space = NormalDistributions(
        sample_dim=request.param, equip=False
    )
    request.cls.random_variable = MultivariateNormalDistributionsRandomVariable(space)


@pytest.mark.usefixtures("general_spaces")
class TestGeneralNormalDistributions(
    InformationManifoldMixinTestCase,
    ProductManifoldTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = GeneralNormalDistributionsTestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(3, 5),
    ],
)
def equipped_general_spaces(request):
    request.cls.space = NormalDistributions(sample_dim=request.param)


@pytest.mark.redundant
@pytest.mark.usefixtures("equipped_general_spaces")
class TestGeneralNormalMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = GeneralNormalMetricTestData()
