import random

import pytest

from geomstats.geometry.stratified.wald_space import (
    NaiveProjectionGeodesicSolver,
    SuccessiveProjectionGeodesicSolver,
    WaldSpace,
    make_topologies,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.stratified.point_set import (
    PointSetMetricTestCase,
    PointSetTestCase,
)
from geomstats.test_cases.geometry.stratified.wald_space import (
    RandomGroveDataGenerator,
    WaldGeodesicSolverTestCase,
    WaldTestCase,
)

from .data.point_set import PointSetTestData, PointTestData
from .data.wald_space import (
    MakePartitionsTestData,
    Wald2TestData,
    Wald3TestData,
    WaldGeodesicSolverTestData,
    WaldSpaceMetricTestData,
)


class TestMakePartitions(TestCase, metaclass=DataBasedParametrizer):
    testing_data = MakePartitionsTestData()

    def test_number_of_topologies(self, n_labels, expected):
        topologies = list(make_topologies(n_labels))

        self.assertTrue(len(topologies) == expected)


class TestWald(WaldTestCase, metaclass=DataBasedParametrizer):
    _n_labels = random.randint(4, 5)
    space = WaldSpace(n_labels=_n_labels, equip=False)

    testing_data = PointTestData()


@pytest.mark.smoke
class TestWald2(WaldTestCase, metaclass=DataBasedParametrizer):
    space = WaldSpace(n_labels=2, equip=False)

    testing_data = Wald2TestData()


@pytest.mark.smoke
class TestWald3(WaldTestCase, metaclass=DataBasedParametrizer):
    space = WaldSpace(n_labels=3, equip=False)

    testing_data = Wald3TestData()


class TestWaldSpace(PointSetTestCase, metaclass=DataBasedParametrizer):
    _n_labels = random.randint(4, 5)
    space = WaldSpace(n_labels=_n_labels, equip=False)

    testing_data = PointSetTestData()


class TestWaldSpaceMetric(PointSetMetricTestCase, metaclass=DataBasedParametrizer):
    _n_labels = random.randint(4, 5)
    space = WaldSpace(n_labels=_n_labels)

    data_generator = RandomGroveDataGenerator(space, space.random_point().topology)
    testing_data = WaldSpaceMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        (NaiveProjectionGeodesicSolver, dict(n_grid=random.randint(5, 10))),
        (SuccessiveProjectionGeodesicSolver, dict(n_grid=random.randint(3, 5) * 2)),
        (SuccessiveProjectionGeodesicSolver, dict(n_grid=random.randint(3, 5) * 2 + 1)),
    ],
)
def geodesic_solvers(request):
    GeodesicSolver, kwargs = request.param

    n_labels = random.randint(4, 5)
    space = WaldSpace(n_labels=n_labels)

    request.cls.geodesic_solver = GeodesicSolver(space, **kwargs)
    request.cls.data_generator = RandomGroveDataGenerator(
        space, space.random_point().topology
    )


@pytest.mark.usefixtures("geodesic_solvers")
class TestWaldGeodesicSolver(
    WaldGeodesicSolverTestCase, metaclass=DataBasedParametrizer
):
    testing_data = WaldGeodesicSolverTestData()
