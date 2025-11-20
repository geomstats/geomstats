from geomstats.geometry.positive_reals import PositiveReals
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.positive_reals import PositiveRealsMetricTestData, PositiveRealsTestData


class TestPositiveReals(VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer):
    space = PositiveReals(equip=False)
    testing_data = PositiveRealsTestData()


class TestPositiveRealsMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = PositiveReals()
    testing_data = PositiveRealsMetricTestData()
