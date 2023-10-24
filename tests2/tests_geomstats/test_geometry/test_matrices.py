import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.general_linear import SquareMatrices
from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.matrices import (
    MatricesMetricTestCase,
    MatricesTestCase,
    MatrixOperationsTestCase,
)

from .data.matrices import (
    MatricesMetric22TestData,
    MatricesMetricTestData,
    MatricesTestData,
    MatrixOperationsSmokeTestData,
    MatrixOperationsTestData,
)


class MatrixOperationsDataGenerator:
    def __init__(self, m_bounds=(2, 4), n_bounds=(2, 4)):
        self.m_bounds = m_bounds
        self.n_bounds = n_bounds

        self._n_shape_trials = 20

    def _random_shape(self):
        return (random.randint(*self.m_bounds), random.randint(*self.n_bounds))

    def _random_square_shape(self):
        return (random.randint(*self.m_bounds),)

    def _random_non_square_shape(self):
        m = random.randint(*self.m_bounds)
        for _ in range(self._n_shape_trials):
            n = random.randint(*self.n_bounds)
            if n != m:
                break

        return (m, n)

    def random_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_shape()

        return Matrices(*shape, equip=False).random_point(n_points)

    def random_square_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_square_shape()

        return SquareMatrices(*shape).random_point(n_points)

    def random_non_square_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_non_square_shape()

        return self.random_mat(n_points, shape)

    def random_symmetric_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_square_shape()

        return SymmetricMatrices(*shape, equip=False).random_point(n_points)

    def random_diagonal_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_square_shape()

        n = shape[0]
        identity = gs.eye(n)
        batch_shape = (n_points,) if n_points > 1 else ()
        coeffs = gs.random.uniform(size=batch_shape + (n,))
        return gs.einsum("...,ij->...ij", coeffs, identity)

    def random_lower_triangular_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_square_shape()

        return LowerTriangularMatrices(*shape).random_point(n_points)

    def random_upper_triangular_mat(self, n_points=1, shape=None):
        return Matrices.transpose(self.random_lower_triangular_mat(n_points, shape))

    def random_spd_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_square_shape()

        return SPDMatrices(*shape).random_point(n_points)

    def random_skew_symmetric_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_square_shape()

        return SkewSymmetricMatrices(*shape).random_point(n_points)


class TestMatrixOperations(MatrixOperationsTestCase, metaclass=DataBasedParametrizer):
    data_generator = MatrixOperationsDataGenerator()
    testing_data = MatrixOperationsTestData()


@pytest.mark.smoke
class TestMatrixOperationsSmokeTestData(
    MatrixOperationsTestCase, metaclass=DataBasedParametrizer
):
    testing_data = MatrixOperationsSmokeTestData()


@pytest.fixture(
    scope="class",
    params=[
        (2, 3),
        (random.randint(3, 5), random.randint(3, 5)),
    ],
)
def spaces(request):
    m, n = request.param
    request.cls.space = Matrices(m=m, n=n, equip=False)


@pytest.mark.usefixtures("spaces")
class TestMatrices(MatricesTestCase, metaclass=DataBasedParametrizer):
    testing_data = MatricesTestData()


@pytest.fixture(
    scope="class",
    params=[
        (2, 3),
        (random.randint(3, 5), random.randint(3, 5)),
    ],
)
def equipped_spaces(request):
    m, n = request.param
    request.cls.space = Matrices(m=m, n=n)


@pytest.mark.usefixtures("equipped_spaces")
class TestMatricesMetric(MatricesMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = MatricesMetricTestData()


@pytest.mark.smoke
class TestMatricesMetric22(MatricesMetricTestCase, metaclass=DataBasedParametrizer):
    space = Matrices(2, 2, equip=False)
    space.equip_with_metric(MatricesMetric)
    testing_data = MatricesMetric22TestData()
