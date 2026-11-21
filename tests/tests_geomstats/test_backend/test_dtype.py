import pytest
from polpo.testing.parametrizers import DataBasedParametrizer

from geomstats.test_cases.backend.dtype import DtypeTestCase

from .data.dtype import DtypeTestData


@pytest.mark.type
class TestDtype(DtypeTestCase, metaclass=DataBasedParametrizer):
    testing_data = DtypeTestData()
