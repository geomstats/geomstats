import random

from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.complex_matrices import ComplexMatrices
from geomstats.test_cases.geometry.base import ComplexMatrixVectorSpaceTestCase
from geomstats.test_cases.geometry.complex_matrices import (
    ComplexMatrixOperationsTestCase,
)
from geomstats.test_cases.geometry.complex_riemannian_metric import (
    ComplexRiemannianMetricTestCase,
)

from .data.base import ComplexMatrixVectorSpaceTestData
from .data.complex_matrices import ComplexMatrixOperationsTestData
from .data.complex_riemannian_metric import ComplexRiemannianMetricTestData


class TestComplexMatrices(
    ComplexMatrixVectorSpaceTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ComplexMatrices(
        m=random.randint(2, 5), n=random.randint(2, 5), equip=False
    )
    testing_data = ComplexMatrixVectorSpaceTestData(space)


class TestComplexMatricesMetric(
    ComplexRiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ComplexMatrices(m=random.randint(2, 5), n=random.randint(2, 5))
    testing_data = ComplexRiemannianMetricTestData(space)


class TestComplexMatrixOperations(
    ComplexMatrixOperationsTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = ComplexMatrixOperationsTestData()
