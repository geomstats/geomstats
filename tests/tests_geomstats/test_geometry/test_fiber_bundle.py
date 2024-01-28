import random

import pytest

from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import MatricesMetric
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.fiber_bundle import (
    FiberBundleTestCase,
    GeneralLinearBuresWassersteinBundle,
)

from .data.fiber_bundle import GeneralLinearBuresWassersteinBundleTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def bundle_spaces(request):
    n = request.param

    request.cls.total_space = total_space = GeneralLinear(n, equip=False)
    total_space.equip_with_metric(MatricesMetric)
    total_space.fiber_bundle = GeneralLinearBuresWassersteinBundle(total_space)

    request.cls.base = SPDMatrices(n=n, equip=False)


@pytest.mark.usefixtures("bundle_spaces")
class TestGeneralLinearBuresWassersteinBundle(
    FiberBundleTestCase, metaclass=DataBasedParametrizer
):
    testing_data = GeneralLinearBuresWassersteinBundleTestData()
