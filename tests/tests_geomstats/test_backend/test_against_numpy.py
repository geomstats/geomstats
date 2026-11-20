from polpo.testing.parametrizers import DataBasedParametrizer

from geomstats.test.test_case import deselect_if_np
from geomstats.test_cases.backend.against_numpy import AgainstNumpyTestCase

from .data.against_numpy import AgainstNumpyTestData


@deselect_if_np
class TestAgainstNumpy(AgainstNumpyTestCase, metaclass=DataBasedParametrizer):
    testing_data = AgainstNumpyTestData()
