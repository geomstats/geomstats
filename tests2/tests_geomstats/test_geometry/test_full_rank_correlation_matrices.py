import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.full_rank_correlation_matrices import (
    CorrelationMatricesBundle,
    FullRankCorrelationAffineQuotientMetric,
    FullRankCorrelationMatrices,
)
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase
from geomstats.test_cases.geometry.full_rank_correlation_matrices import (
    FullRankCorrelationMatricesTestCase,
)
from geomstats.test_cases.geometry.quotient_metric import QuotientMetricTestCase

from .data.full_rank_correlation_matrices import (
    CorrelationMatricesBundleTestData,
    FullRankCorrelationAffineQuotientMetricTestData,
    FullRankCorrelationMatricesTestData,
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
    request.cls.bundle = CorrelationMatricesBundle(total_space)
    request.cls.base = FullRankCorrelationMatrices(n=n, equip=False)


@pytest.mark.usefixtures("bundles")
class TestCorrelationMatricesBundle(
    FiberBundleTestCase, metaclass=DataBasedParametrizer
):
    testing_data = CorrelationMatricesBundleTestData()

    def test_horizontal_projection_is_horizontal_v2(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        horizontal_vec = self.bundle.horizontal_projection(tangent_vec, base_point)

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
    request.cls.space = space = FullRankCorrelationMatrices(n=n, equip=False)
    space.equip_with_metric(FullRankCorrelationAffineQuotientMetric)


@pytest.mark.redundant
@pytest.mark.usefixtures("affine_quotient_equipped_spaces")
class TestFullRankCorrelationAffineQuotientMetric(
    QuotientMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = FullRankCorrelationAffineQuotientMetricTestData()
