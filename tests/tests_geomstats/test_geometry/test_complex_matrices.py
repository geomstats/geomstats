import random

from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.complex_matrices import ComplexMatrices
from geomstats.test_cases.geometry.base import MatrixVectorSpaceTestCase
from geomstats.test_cases.geometry.complex_matrices import (
    ComplexMatrixOperationsTestCase,
)
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.base import MatrixVectorSpaceTestData
from .data.complex_matrices import ComplexMatrixOperationsTestData
from .data.riemannian_metric import RiemannianMetricTestData


class TestComplexMatrices(
    MatrixVectorSpaceTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ComplexMatrices(m=random.randint(2, 5), n=random.randint(2, 5), equip=False)
    testing_data = MatrixVectorSpaceTestData(space)


class TestComplexMatricesMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ComplexMatrices(m=random.randint(2, 5), n=random.randint(2, 5))
    testing_data = RiemannianMetricTestData(space)


class TestComplexMatrixOperations(
    ComplexMatrixOperationsTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = ComplexMatrixOperationsTestData()
