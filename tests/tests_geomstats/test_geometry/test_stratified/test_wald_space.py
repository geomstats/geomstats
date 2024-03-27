import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.stratified.wald_space import (
    NaiveProjectionGeodesicSolver,
    SuccessiveProjectionGeodesicSolver,
    WaldSpace,
    _squared_dist_and_grad_autodiff,
    make_topologies,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase, autodiff_only
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
    SquaredDistAndGradTestData,
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


@autodiff_only
class TestSquaredDistAndGrad(TestCase, metaclass=DataBasedParametrizer):
    _n_labels = random.randint(4, 5)
    space = WaldSpace(n_labels=_n_labels, equip=False)
    testing_data = SquaredDistAndGradTestData()

    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    @pytest.mark.random
    def test_value_and_grad_against_autodiff(
        self, squared_dist_and_grad, AmbientMetric, atol
    ):
        topology = self.data_generator.random_point().topology

        point = self.space.random_grove_point(topology, n_samples=1)
        ambient_point = self.space.lift(point)

        value_and_grad = squared_dist_and_grad(self.space, topology, ambient_point)
        value_and_grad_autodiff = _squared_dist_and_grad_autodiff(
            self.space, topology, ambient_point
        )

        weights = gs.random.uniform(size=(topology.n_splits,))

        self.space.ambient_space.equip_with_metric(AmbientMetric)
        value, grad = value_and_grad(weights)
        value_autodiff, grad_autodiff = value_and_grad_autodiff(weights)

        self.assertAllClose(value, value_autodiff, atol=atol)
        self.assertAllClose(grad, grad_autodiff, atol=atol)


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
