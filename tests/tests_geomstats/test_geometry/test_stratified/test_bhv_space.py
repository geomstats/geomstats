import random

import pytest

from geomstats.geometry.stratified.bhv_space import TreeSpace
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import np_only
from geomstats.test_cases.geometry.stratified.point_set import (
    PointSetMetricTestCase,
    PointSetTestCase,
    PointTestCase,
)

from .data.bhv_space import BHVMetric5TestData
from .data.point_set import PointMetricTestData, PointSetTestData, PointTestData


@np_only
class TestTree(PointTestCase, metaclass=DataBasedParametrizer):
    _n_labels = random.randint(4, 5)
    space = TreeSpace(n_labels=_n_labels, equip=False)

    testing_data = PointTestData()


@np_only
class TestTreeSpace(PointSetTestCase, metaclass=DataBasedParametrizer):
    _n_labels = random.randint(4, 5)
    space = TreeSpace(n_labels=_n_labels, equip=False)

    testing_data = PointSetTestData()


@np_only
class TestBHVMetric(PointSetMetricTestCase, metaclass=DataBasedParametrizer):
    _n_labels = random.randint(4, 5)
    space = TreeSpace(n_labels=_n_labels, equip=True)

    testing_data = PointMetricTestData()


@pytest.mark.smoke
@np_only
class TestBHVMetric5(PointSetMetricTestCase, metaclass=DataBasedParametrizer):
    space = TreeSpace(n_labels=5, equip=True)

    testing_data = BHVMetric5TestData()

    def test_geodesic(self, initial_point, end_point, t, expected, atol):
        geod_func = self.space.metric.geodesic(initial_point, end_point)
        geod_points = geod_func(t)

        for geod_point, expected_point in zip(geod_points, expected):
            self.assertTrue(geod_point.equal(expected_point, atol))
