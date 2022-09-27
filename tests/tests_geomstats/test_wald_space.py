from tests.conftest import Parametrizer, np_backend
from tests.data.wald_space_data import WaldSpaceTestData, WaldTestData
from tests.stratified_test_cases import PointSetTestCase, PointTestCase

IS_NOT_NP = not np_backend()


class TestWaldSpace(PointSetTestCase, metaclass=Parametrizer):
    skip_all = IS_NOT_NP

    testing_data = WaldSpaceTestData()


class TestWald(PointTestCase, metaclass=Parametrizer):
    skip_all = IS_NOT_NP

    testing_data = WaldTestData()
    _Point = testing_data._Point
