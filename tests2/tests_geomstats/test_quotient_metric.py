import random

import pytest

from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import SPDBuresWassersteinMetric, SPDMatrices
from geomstats.test.comparison import RiemannianMetricComparisonTestCase
from geomstats.test.geometry.fiber_bundle import GeneralLinearBuresWassersteinBundle
from geomstats.test.geometry.spd_matrices import SPDRandomDataGenerator
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.quotient_metric_data import SPDBuresWassersteinQuotientMetricTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spd_with_quotient_metric(request):
    n = request.param
    space = SPDMatrices(n=n, equip=False)
    request.cls.space = space
    space.equip_with_metric(SPDBuresWassersteinMetric)

    other_space = SPDMatrices(n=n, equip=False)
    request.cls.other_space = other_space
    bundle = GeneralLinearBuresWassersteinBundle(n)
    other_space.equip_with_metric(QuotientMetric, fiber_bundle=bundle)

    request.cls.data_generator = SPDRandomDataGenerator(space, amplitude=8.0)


@pytest.mark.usefixtures("spd_with_quotient_metric")
class TestSPDBuresWassersteinQuotientMetric(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDBuresWassersteinQuotientMetricTestData()
