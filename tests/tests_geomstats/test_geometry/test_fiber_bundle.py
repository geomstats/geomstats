import random

from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import MatricesMetric
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.fiber_bundle import (
    FiberBundleTestCase,
    GeneralLinearBuresWassersteinBundle,
)

from .data.fiber_bundle import GeneralLinearBuresWassersteinBundleTestData


class TestGeneralLinearBuresWassersteinBundle(
    FiberBundleTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 5)

    total_space = GeneralLinear(_n, equip=False)
    total_space.equip_with_metric(MatricesMetric)

    total_space.fiber_bundle = GeneralLinearBuresWassersteinBundle(total_space)

    base = SPDMatrices(n=_n, equip=False)

    testing_data = GeneralLinearBuresWassersteinBundleTestData()
