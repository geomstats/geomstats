import pytest

from geomstats.geometry.heisenberg import HeisenbergVectors
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.heisenberg import HeisenbergVectorsTestCase

from .data.heisenberg import HeisenbergVectors3TestData, HeisenbergVectorsTestData


class TestHeisenbergVectors(HeisenbergVectorsTestCase, metaclass=DataBasedParametrizer):
    space = HeisenbergVectors(equip=False)
    testing_data = HeisenbergVectorsTestData()


@pytest.mark.smoke
class TestHeisenbergVectors3(
    HeisenbergVectorsTestCase, metaclass=DataBasedParametrizer
):
    space = HeisenbergVectors(equip=False)
    testing_data = HeisenbergVectors3TestData()
