from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.backend import AgainstScipyTestCase

from .data.against_scipy import AgainstScipyTestData


class TestAgainstScipy(AgainstScipyTestCase, metaclass=DataBasedParametrizer):
    testing_data = AgainstScipyTestData()
