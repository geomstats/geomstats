from geomstats.geometry.heisenberg import HeisenbergVectors
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.heisenberg import HeisenbergVectorsTestCase

from .data.heisenberg import HeisenbergVectorsTestData


class TestHeisenbergVectors(HeisenbergVectorsTestCase, metaclass=DataBasedParametrizer):
    space = HeisenbergVectors()
    testing_data = HeisenbergVectorsTestData()
