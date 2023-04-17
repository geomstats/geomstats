from geomstats.geometry.heisenberg import HeisenbergVectors
from geomstats.test.geometry.heisenberg import HeisenbergVectorsTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.heisenberg_data import HeisenbergVectorsTestData


class TestHeisenbergVectors(HeisenbergVectorsTestCase, metaclass=DataBasedParametrizer):
    space = HeisenbergVectors()
    testing_data = HeisenbergVectorsTestData()
