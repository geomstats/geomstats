from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.backend import BackendTestCase

from .data.backend import BackendTestData


class TestBackend(BackendTestCase, metaclass=DataBasedParametrizer):
    testing_data = BackendTestData()
