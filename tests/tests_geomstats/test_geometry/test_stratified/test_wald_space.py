import random

import pytest

from geomstats.geometry.stratified.wald_space import WaldSpace
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import np_only
from geomstats.test_cases.geometry.stratified.point_set import PointSetTestCase
from geomstats.test_cases.geometry.stratified.wald_space import WaldTestCase

from .data.point_set import PointSetTestData, PointTestData
from .data.wald_space import Wald2TestData, Wald3TestData

# TODO: also us smoke data


@np_only
class TestWald(WaldTestCase, metaclass=DataBasedParametrizer):
    _n_labels = random.randint(4, 5)
    space = WaldSpace(n_labels=_n_labels, equip=False)

    testing_data = PointTestData()


@pytest.mark.smoke
@np_only
class TestWald2(WaldTestCase, metaclass=DataBasedParametrizer):
    space = WaldSpace(n_labels=2, equip=False)

    testing_data = Wald2TestData()


@pytest.mark.smoke
@np_only
class TestWald3(WaldTestCase, metaclass=DataBasedParametrizer):
    space = WaldSpace(n_labels=3, equip=False)

    testing_data = Wald3TestData()


@np_only
class TestWaldSpace(PointSetTestCase, metaclass=DataBasedParametrizer):
    _n_labels = random.randint(4, 5)
    space = WaldSpace(n_labels=_n_labels, equip=False)

    testing_data = PointSetTestData()
