import random

import pytest

from geomstats.geometry.spd_matrices import SPDBuresWassersteinMetric, SPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
)
from geomstats.variants.geometry.spd_matrices import QuotientSPDBuresWassersteinMetric

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
    other_space.equip_with_metric(QuotientSPDBuresWassersteinMetric)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=10.0)


@pytest.mark.usefixtures("spd_with_quotient_metric")
class TestSPDBuresWassersteinQuotientMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDBuresWassersteinQuotientMetricCmpTestData()
