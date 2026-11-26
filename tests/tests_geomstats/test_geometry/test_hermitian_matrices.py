import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.hermitian_matrices import HermitianMatrices
from geomstats.test_cases.geometry.base import MatrixVectorSpaceTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.base import MatrixVectorSpaceTestData
from .data.hermitian_matrices import HermitianMatrices2TestData
from .data.riemannian_metric import RiemannianMetricTestData


class TestHermitianMatrices(
    MatrixVectorSpaceTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = HermitianMatrices(n=random.randint(2, 5), equip=False)
    testing_data = MatrixVectorSpaceTestData(space)


@pytest.mark.smoke
class TestHermitianMatrices2(
    MatrixVectorSpaceTestCase, metaclass=DataBasedParametrizer
):
    space = HermitianMatrices(n=2, equip=False)
    testing_data = HermitianMatrices2TestData()


@pytest.mark.redundant
class TestComplexMatricesMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = HermitianMatrices(n=random.randint(2, 5))
    testing_data = RiemannianMetricTestData(space)
