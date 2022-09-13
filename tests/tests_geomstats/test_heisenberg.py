"""Unit tests for the 3D heisenberg group in vector representation."""

import geomstats.backend as gs
from tests.conftest import Parametrizer
from tests.data.heisenberg_data import HeisenbergVectorsTestData
from tests.geometry_test_cases import LieGroupTestCase, VectorSpaceTestCase


class TestHeisenbergVectors(
    LieGroupTestCase, VectorSpaceTestCase, metaclass=Parametrizer
):

    testing_data = HeisenbergVectorsTestData()

    def test_dimension(self, expected):
        self.assertAllClose(self.Space().dim, expected)

    def test_jacobian_translation(self, vec, expected):
        self.assertAllClose(
            self.Space().jacobian_translation(gs.array(vec)), gs.array(expected)
        )

    def test_is_tangent(self, vector, expected):
        group = self.Space()
        result = group.is_tangent(gs.array(vector))
        self.assertAllClose(result, gs.array(expected))
