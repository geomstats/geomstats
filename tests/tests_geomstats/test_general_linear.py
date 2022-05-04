"""Unit tests for the General Linear group."""

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear, SquareMatrices
from tests.conftest import Parametrizer
from tests.data.general_linear_data import GeneralLinearTestData, SquareMatricesTestData
from tests.geometry_test_cases import (
    LieGroupTestCase,
    MatrixLieAlgebraTestCase,
    OpenSetTestCase,
)


class TestGeneralLinear(LieGroupTestCase, OpenSetTestCase, metaclass=Parametrizer):
    space = group = GeneralLinear
    skip_test_log_after_exp = True
    skip_test_exp_after_log = True

    testing_data = GeneralLinearTestData()

    def test_belongs(self, n, point, expected):
        group = self.space(n)
        self.assertAllClose(group.belongs(gs.array(point)), gs.array(expected))

    def test_compose(self, n, mat1, mat2, expected):
        group = self.space(n)
        self.assertAllClose(
            group.compose(gs.array(mat1), gs.array(mat2)), gs.array(expected)
        )

    def test_inv(self, n, mat, expected):
        group = self.space(n)
        self.assertAllClose(group.inverse(gs.array(mat)), gs.array(expected))

    def test_exp(self, n, tangent_vec, base_point, expected):
        group = self.space(n)
        expected = gs.cast(gs.array(expected), gs.float64)
        tangent_vec = gs.cast(gs.array(tangent_vec), gs.float64)
        base_point = (
            None if base_point is None else gs.cast(gs.array(base_point), gs.float64)
        )
        self.assertAllClose(group.exp(tangent_vec, base_point), gs.array(expected))

    def test_log(self, n, point, base_point, expected):
        group = self.space(n)
        expected = gs.cast(gs.array(expected), gs.float64)
        point = gs.cast(gs.array(point), gs.float64)
        base_point = (
            None if base_point is None else gs.cast(gs.array(base_point), gs.float64)
        )
        self.assertAllClose(group.log(point, base_point), expected)

    def test_orbit(self, n, point, base_point, time, expected):
        group = self.space(n)
        result = group.orbit(gs.array(point), gs.array(base_point))(time)
        self.assertAllClose(result, gs.array(expected))


class TestSquareMatrices(MatrixLieAlgebraTestCase, metaclass=Parametrizer):
    space = algebra = SquareMatrices

    testing_data = SquareMatricesTestData()

    def test_belongs(self, n, mat, expected):
        space = self.space(n)
        self.assertAllClose(space.belongs(gs.array(mat)), gs.array(expected))
