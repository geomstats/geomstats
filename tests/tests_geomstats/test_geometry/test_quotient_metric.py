import random

import pytest

from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import MatricesMetric
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import SPDBuresWassersteinMetric, SPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.fiber_bundle import (
    GeneralLinearBuresWassersteinBundle,
)
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
)

from .data.quotient_metric import SPDBuresWassersteinQuotientMetricCmpTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spd_with_quotient_metric(request):
    n = request.param
    request.cls.space = space = SPDMatrices(n=n, equip=False)
    space.equip_with_metric(SPDBuresWassersteinMetric)

    request.cls.other_space = other_space = SPDMatrices(n=n, equip=False)

    total_space = GeneralLinear(n=n, equip=False)
    total_space.equip_with_metric(MatricesMetric)
    total_space.fiber_bundle = GeneralLinearBuresWassersteinBundle(total_space)
    other_space.equip_with_metric(QuotientMetric, total_space=total_space)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=10.0)


@pytest.mark.usefixtures("spd_with_quotient_metric")
class TestSPDBuresWassersteinQuotientMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDBuresWassersteinQuotientMetricCmpTestData()
