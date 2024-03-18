from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.scalar_product_metric import (
    CustomizationTestCase,
    InstantiationTestCase,
    WrapperTestCase,
)

from .data.scalar_product_metric import (
    CustomizationTestData,
    InstantiationTestData,
    WrapperTestData,
)


class TestWrapper(WrapperTestCase, metaclass=DataBasedParametrizer):
    testing_data = WrapperTestData()


class TestInstantiation(InstantiationTestCase, metaclass=DataBasedParametrizer):
    space = Euclidean(dim=3)
    testing_data = InstantiationTestData()


class TestCustomization(CustomizationTestCase, metaclass=DataBasedParametrizer):
    Metric = EuclideanMetric
    space = Euclidean(dim=3)
    testing_data = CustomizationTestData()
