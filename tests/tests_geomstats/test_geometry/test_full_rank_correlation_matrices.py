import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.diffeo import ComposedDiffeo
from geomstats.geometry.full_rank_correlation_matrices import (
    CorrelationMatricesBundle,
    FullRankCorrelationMatrices,
    PolyHyperbolicCholeskyMetric,
)
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.open_hemisphere import (
    OpenHemispheresProduct,
    OpenHemisphereToHyperboloidDiffeo,
)
from geomstats.geometry.positive_lower_triangular_matrices import (
    UnitNormedRowsPLTDiffeo,
    UnitNormedRowsPLTMatrices,
)
from geomstats.geometry.spd_matrices import CholeskyMap, SPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase
from geomstats.test_cases.geometry.full_rank_correlation_matrices import (
    FullRankCorrelationMatricesTestCase,
)
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase
from geomstats.test_cases.geometry.quotient_metric import QuotientMetricTestCase

from .data.diffeo import DiffeoTestData
from .data.full_rank_correlation_matrices import (
    CorrelationMatricesBundleTestData,
    FullRankCorrelationAffineQuotientMetricTestData,
    FullRankCorrelationMatricesTestData,
    PolyHyperbolicCholeskyMetricTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        3,
        random.randint(4, 8),
    ],
)
def spaces(request):
    request.cls.space = FullRankCorrelationMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestFullRankCorrelationMatrices(
    FullRankCorrelationMatricesTestCase, metaclass=DataBasedParametrizer
):
    testing_data = FullRankCorrelationMatricesTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def bundles(request):
    n = request.param
    request.cls.total_space = total_space = SPDMatrices(n=n, equip=True)
    total_space.fiber_bundle = CorrelationMatricesBundle(total_space)
    request.cls.base = FullRankCorrelationMatrices(n=n, equip=False)


@pytest.mark.usefixtures("bundles")
class TestCorrelationMatricesBundle(
    FiberBundleTestCase, metaclass=DataBasedParametrizer
):
    testing_data = CorrelationMatricesBundleTestData()

    def test_horizontal_projection_is_horizontal_v2(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        horizontal_vec = self.total_space.fiber_bundle.horizontal_projection(
            tangent_vec, base_point
        )

        inverse = GeneralLinear.inverse(base_point)
        product_1 = Matrices.mul(horizontal_vec, inverse)
        product_2 = Matrices.mul(inverse, horizontal_vec)
        is_horizontal = gs.all(
            self.base.is_tangent(product_1 + product_2, base_point, atol=atol)
        )

        self.assertTrue(is_horizontal)


@pytest.fixture(
    scope="class",
    params=[
        random.randint(3, 5),
    ],
)
def affine_quotient_equipped_spaces(request):
    n = request.param
    request.cls.space = FullRankCorrelationMatrices(n=n)


@pytest.mark.redundant
@pytest.mark.usefixtures("affine_quotient_equipped_spaces")
class TestFullRankCorrelationAffineQuotientMetric(
    QuotientMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = FullRankCorrelationAffineQuotientMetricTestData()


class TestCholeskyMap(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)
    space = FullRankCorrelationMatrices(n=_n, equip=False)
    image_space = UnitNormedRowsPLTMatrices(n=_n, equip=False)
    diffeo = CholeskyMap()
    testing_data = DiffeoTestData()


class TestDiffeoToOpenHemispheres(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(3, 5)
    space = FullRankCorrelationMatrices(n=_n, equip=False)
    image_space = OpenHemispheresProduct(n=_n, equip=False)
    _diffeos = [CholeskyMap(), UnitNormedRowsPLTDiffeo(_n)]
    diffeo = ComposedDiffeo(_diffeos)
    testing_data = DiffeoTestData()


class TestDiffeoToHyperboloid(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = 2
    space = FullRankCorrelationMatrices(n=_n, equip=False)
    image_space = Hyperboloid(dim=1, equip=False)
    _diffeos = [
        CholeskyMap(),
        UnitNormedRowsPLTDiffeo(_n),
        OpenHemisphereToHyperboloidDiffeo(),
    ]
    diffeo = ComposedDiffeo(_diffeos)
    testing_data = DiffeoTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def phc_equipped_spaces(request):
    n = request.param
    request.cls.space = FullRankCorrelationMatrices(n, equip=False).equip_with_metric(
        PolyHyperbolicCholeskyMetric,
    )


@pytest.mark.usefixtures("phc_equipped_spaces")
class TestPolyHyperbolicCholeskyMetric(
    PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = PolyHyperbolicCholeskyMetricTestData()
