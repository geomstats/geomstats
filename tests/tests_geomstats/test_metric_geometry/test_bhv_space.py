import random

import pytest
from polpo.testing.parametrizers import DataBasedParametrizer

from geomstats.metric_geometry.bhv_space import TreeSpace
from geomstats.test.test_case import np_only
from geomstats.test_cases.metric_geometry.bhv_space import (
    BHVMetricTestCase,
    TreeTestCase,
    TreeTopologyTestCase,
)
from geomstats.test_cases.metric_geometry.point_set import (
    PointSetTestCase,
    PointTestCase,
)

from .data.bhv_space import (
    BHVMetric5TestData,
    BHVMetricTestData,
    TreeTestData,
    TreeTopologyTestData,
)
from .data.point_set import PointSetTestData, PointTestData


@pytest.mark.smoke
class TestTreeTopology(TreeTopologyTestCase, metaclass=DataBasedParametrizer):
    testing_data = TreeTopologyTestData()


@pytest.mark.smoke
class TestTree(TreeTestCase, metaclass=DataBasedParametrizer):
    testing_data = TreeTestData()


class TestTreeAsPoint(PointTestCase, metaclass=DataBasedParametrizer):
    space = TreeSpace(n_labels=random.randint(4, 5), equip=False)
    testing_data = PointTestData(space)


class TestTreeSpace(PointSetTestCase, metaclass=DataBasedParametrizer):
    space = TreeSpace(n_labels=random.randint(4, 5), equip=False)
    testing_data = PointSetTestData(space)


@pytest.mark.smoke
@np_only
class TestBHVMetric5(BHVMetricTestCase, metaclass=DataBasedParametrizer):
    space = TreeSpace(n_labels=5)
    testing_data = BHVMetric5TestData()


@np_only
class TestBHVMetric(BHVMetricTestCase, metaclass=DataBasedParametrizer):
    space = TreeSpace(n_labels=random.randint(4, 5))
    testing_data = BHVMetricTestData(space)
