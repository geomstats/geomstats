import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.hermitian_matrices import HermitianMatrices
from geomstats.test_cases.geometry.base import ComplexMatrixVectorSpaceTestCase
from geomstats.test_cases.geometry.complex_riemannian_metric import (
    ComplexRiemannianMetricTestCase,
)

from .data.base import ComplexMatrixVectorSpaceTestData
from .data.complex_riemannian_metric import ComplexRiemannianMetricTestData
from .data.hermitian_matrices import HermitianMatrices2TestData


@pytest.mark.smoke
class TestHermitianMatrices2(
    ComplexMatrixVectorSpaceTestCase, metaclass=DataBasedParametrizer
):
    space = HermitianMatrices(n=2, equip=False)
    testing_data = HermitianMatrices2TestData()


class TestHermitianMatrices(
    ComplexMatrixVectorSpaceTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = HermitianMatrices(n=random.randint(2, 5), equip=False)
    testing_data = ComplexMatrixVectorSpaceTestData(space)


@pytest.mark.redundant
class TestComplexMatricesMetric(
    ComplexRiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = HermitianMatrices(n=random.randint(2, 5))
    testing_data = ComplexRiemannianMetricTestData(space)
