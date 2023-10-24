from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.backend import DtypeTestCase

from .data.dtype import DtypeTestData


class TestDtype(DtypeTestCase, metaclass=DataBasedParametrizer):
    testing_data = DtypeTestData()
