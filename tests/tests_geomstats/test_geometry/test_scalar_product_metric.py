from geomstats.geometry.euclidean import Euclidean
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.scalar_product_metric import (
    InstantiationTestCase,
    WrapperTestCase,
)

from .data.scalar_product_metric import InstantiationTestData, WrapperTestData


class TestWrapper(WrapperTestCase, metaclass=DataBasedParametrizer):
    testing_data = WrapperTestData()


class TestInstantiation(InstantiationTestCase, metaclass=DataBasedParametrizer):
    space = Euclidean(dim=3)
    testing_data = InstantiationTestData()
