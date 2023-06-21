from geomstats.geometry.positive_reals import PositiveReals
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.positive_reals import (
    PositiveRealsMetricTestCase,
    PositiveRealsTestCase,
)
from tests2.tests_geomstats.test_geometry.data.positive_reals import (
    PositiveRealsMetricTestData,
)

from .data.positive_reals import PositiveRealsTestData


class TestPositiveReals(PositiveRealsTestCase, metaclass=DataBasedParametrizer):
    space = PositiveReals(equip=False)
    testing_data = PositiveRealsTestData()


class TestPositiveRealsMetric(
    PositiveRealsMetricTestCase, metaclass=DataBasedParametrizer
):
    space = PositiveReals(equip=True)
    testing_data = PositiveRealsMetricTestData()
