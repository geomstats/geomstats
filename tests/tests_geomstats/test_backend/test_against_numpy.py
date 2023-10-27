from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.backend import AgainstNumpyTestCase

from .data.against_numpy import AgainstNumpyTestData


class TestAgainstNumpy(AgainstNumpyTestCase, metaclass=DataBasedParametrizer):
    testing_data = AgainstNumpyTestData()
