from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.algebra_utils import AlgebraUtilsTestCase

from .data.algebra_utils import AlgebraUtilsTestData


class TestAlgebraUtils(AlgebraUtilsTestCase, metaclass=DataBasedParametrizer):
    testing_data = AlgebraUtilsTestData()
