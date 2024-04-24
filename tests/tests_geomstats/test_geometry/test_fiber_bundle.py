import random

from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.group_action import SpecialOrthogonalComposeAction
from geomstats.geometry.matrices import MatricesMetric
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase

from .data.fiber_bundle import GeneralLinearBuresWassersteinBundleTestData


class TestGeneralLinearBuresWassersteinBundle(
    FiberBundleTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 5)

    total_space = GeneralLinear(_n, equip=False)

    total_space.equip_with_metric(MatricesMetric)
    total_space.equip_with_group_action(SpecialOrthogonalComposeAction(total_space.n))
    total_space.equip_with_quotient_structure()

    base = SPDMatrices(n=_n, equip=False)

    testing_data = GeneralLinearBuresWassersteinBundleTestData()
