from tests.conftest import Parametrizer, np_backend
from tests.data.bhv_space_data import BHVMetricTestData, TreeSpaceTestData, TreeTestData
from tests.stratified_test_cases import (
    PointSetMetricTestCase,
    PointSetTestCase,
    PointTestCase,
)

IS_NOT_NP = not np_backend()


class TestTree(PointTestCase, metaclass=Parametrizer):
    skip_all = IS_NOT_NP

    testing_data = TreeTestData()
    _Point = testing_data._Point


class TestTreeSpace(PointSetTestCase, metaclass=Parametrizer):
    skip_all = IS_NOT_NP

    testing_data = TreeSpaceTestData()
    _Point = testing_data._Point


class TestBHVMetric(PointSetMetricTestCase, metaclass=Parametrizer):
    skip_all = IS_NOT_NP

    testing_data = BHVMetricTestData()
