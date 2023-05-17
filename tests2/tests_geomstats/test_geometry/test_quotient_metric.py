import random

import pytest

from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import MatricesMetric
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import SPDBuresWassersteinMetric, SPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.comparison import RiemannianMetricComparisonTestCase
from geomstats.test_cases.geometry.fiber_bundle import (
    GeneralLinearBuresWassersteinBundle,
)

from .data.quotient_metric import SPDBuresWassersteinQuotientMetricTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        # random.randint(3, 5),
    ],
)
def spd_with_quotient_metric(request):
    n = request.param
    request.cls.space = space = SPDMatrices(n=n, equip=False)
    space.equip_with_metric(SPDBuresWassersteinMetric)

    request.cls.other_space = other_space = SPDMatrices(n=n, equip=False)

    total_space = GeneralLinear(n=n, equip=False)
    total_space.equip_with_metric(MatricesMetric)

    bundle = GeneralLinearBuresWassersteinBundle(total_space)
    other_space.equip_with_metric(QuotientMetric, fiber_bundle=bundle)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=10.0)


@pytest.mark.usefixtures("spd_with_quotient_metric")
class TestSPDBuresWassersteinQuotientMetric(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDBuresWassersteinQuotientMetricTestData()
