import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import (
    BasisRepresentationDiffeo,
    FlattenDiffeo,
    Matrices,
    MatricesDiagMetric,
    MatricesMetric,
)
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.matrices import (
    MatricesMetricTestCase,
    MatricesTestCase,
    MatrixOperationsTestCase,
)

from .data.diffeo import DiffeoTestData
from .data.matrices import (
    MatricesMetric22TestData,
    MatricesMetricTestData,
    MatricesTestData,
    MatrixOperationsTestData,
)


def _get_random_shape(low=2, high=5):
    """Get a random matrix shape.

    Parameters
    ----------
    low : int
        Minimum number of rows and columns.
    high : int
        Maximum number of rows and columns.

    Returns
    -------
    m, n : int
        Number of rows and columns.
    """
    return random.randint(low, high), random.randint(low, high)


class TestFlattenDiffeo(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    _m, _n = _get_random_shape(low=3)

    domain_space = Matrices(m=_m, n=_n, equip=False)
    image_space = Euclidean(dim=_m * _n, equip=False)
    diffeo = FlattenDiffeo(_m, _n)

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestBasisRepresentationDiffeo(
    DiffeoTestCase, metaclass=GeometricDataBasedParametrizer
):
    _m, _n = _get_random_shape(low=3)

    domain_space = Matrices(m=_m, n=_n, equip=False)
    image_space = Euclidean(dim=_m * _n, equip=False)
    diffeo = BasisRepresentationDiffeo(domain_space)

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestMatrices(MatricesTestCase, metaclass=GeometricDataBasedParametrizer):
    _m, _n = _get_random_shape()
    space = Matrices(m=_m, n=_n, equip=False)
    testing_data = MatricesTestData(space)


class TestMatricesMetric(
    MatricesMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    _m, _n = _get_random_shape()
    space = Matrices(m=_m, n=_n)
    testing_data = MatricesMetricTestData(space)


@pytest.mark.smoke
class TestMatricesMetric22(MatricesMetricTestCase, metaclass=DataBasedParametrizer):
    space = Matrices(2, 2, equip=False)
    space.equip_with_metric(MatricesMetric)
    testing_data = MatricesMetric22TestData()


class TestMatricesDiagMetric(
    MatricesMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    _m, _n = _get_random_shape()
    space = Matrices(m=_m, n=_n, equip=False).equip_with_metric(MatricesDiagMetric)
    testing_data = MatricesMetricTestData(space)


class TestMatrixOperations(
    MatrixOperationsTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = MatrixOperationsTestData()
