import random

import pytest

from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.geometry.fiber_bundle import (
    GeneralLinearBuresWassersteinBundle,
    GeneralLinearBuresWassersteinBundleTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.fiber_bundle_data import GeneralLinearBuresWassersteinBundleTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def bundle_spaces(request):
    n = request.param
    request.cls.space = GeneralLinearBuresWassersteinBundle(n)
    request.cls.base = SPDMatrices(n=n, equip=False)


@pytest.mark.usefixtures("bundle_spaces")
class TestGeneralLinearBuresWassersteinBundle(
    GeneralLinearBuresWassersteinBundleTestCase, metaclass=DataBasedParametrizer
):
    testing_data = GeneralLinearBuresWassersteinBundleTestData()
