import pytest

from geomstats.geometry.heisenberg import HeisenbergVectors, HeisenbergVectorsToMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import VectorSpaceTestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.lie_group import LieGroupTestCase

from .data.heisenberg import (
    HeisenbergVectors3TestData,
    HeisenbergVectorsTestData,
    HeisenbergVectorsToMatricesSmokeTestData,
    HeisenbergVectorsToMatricesTestData,
)


class TestHeisenbergVectorsToMatrices(DiffeoTestCase, metaclass=DataBasedParametrizer):
    space = HeisenbergVectors(equip=False)
    diffeo = HeisenbergVectorsToMatrices()
    testing_data = HeisenbergVectorsToMatricesTestData()


@pytest.mark.smoke
class TestHeisenbergVectorsToMatricesSmoke(
    DiffeoTestCase, metaclass=DataBasedParametrizer
):
    diffeo = HeisenbergVectorsToMatrices()
    testing_data = HeisenbergVectorsToMatricesSmokeTestData()


class TestHeisenbergVectors(
    LieGroupTestCase, VectorSpaceTestCase, metaclass=DataBasedParametrizer
):
    space = HeisenbergVectors(equip=False)
    testing_data = HeisenbergVectorsTestData()


@pytest.mark.smoke
class TestHeisenbergVectors3(
    LieGroupTestCase, VectorSpaceTestCase, metaclass=DataBasedParametrizer
):
    space = HeisenbergVectors(equip=False)
    testing_data = HeisenbergVectors3TestData()
