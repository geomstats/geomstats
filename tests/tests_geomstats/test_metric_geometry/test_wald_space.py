import random

import pytest
from polpo.testing.parametrizers import DataBasedParametrizer

from geomstats.metric_geometry.wald_space import (
    NaiveProjectionGeodesicSolver,
    SuccessiveProjectionGeodesicSolver,
    WaldSpace,
)
from geomstats.test.random import RandomGroveDataGenerator
from geomstats.test.test_case import torch_only
from geomstats.test_cases.metric_geometry.point_set import (
    PointSetMetricTestCase,
    PointSetTestCase,
)
from geomstats.test_cases.metric_geometry.wald_space import (
    MakeTopologiesTestCase,
    SquaredDistAndGradTestCase,
    WaldGeodesicSolverTestCase,
    WaldTestCase,
)

from .data.point_set import PointSetMetricTestData, PointSetTestData, PointTestData
from .data.wald_space import (
    MakePartitionsTestData,
    SquaredDistAndGradTestData,
    Wald2TestData,
    Wald3TestData,
    WaldGeodesicSolverTestData,
)


@pytest.mark.smoke
class TestMakePartitions(MakeTopologiesTestCase, metaclass=DataBasedParametrizer):
    testing_data = MakePartitionsTestData()


@torch_only
class TestSquaredDistAndGrad(
    SquaredDistAndGradTestCase, metaclass=DataBasedParametrizer
):
    space = WaldSpace(n_labels=random.randint(4, 5), equip=False)
    testing_data = SquaredDistAndGradTestData()


@pytest.mark.smoke
class TestWald2(WaldTestCase, metaclass=DataBasedParametrizer):
    space = WaldSpace(n_labels=2, equip=False)
    testing_data = Wald2TestData()


@pytest.mark.smoke
class TestWald3(WaldTestCase, metaclass=DataBasedParametrizer):
    space = WaldSpace(n_labels=3, equip=False)
    testing_data = Wald3TestData()


class TestWald(WaldTestCase, metaclass=DataBasedParametrizer):
    space = WaldSpace(n_labels=random.randint(4, 5), equip=False)
    testing_data = PointTestData(space)


class TestWaldSpace(PointSetTestCase, metaclass=DataBasedParametrizer):
    space = WaldSpace(n_labels=random.randint(4, 5), equip=False)
    testing_data = PointSetTestData(space)


class TestWaldSpaceMetric(PointSetMetricTestCase, metaclass=DataBasedParametrizer):
    space = WaldSpace(n_labels=random.randint(4, 5))
    testing_data = PointSetMetricTestData(
        space,
        method_defaults={
            "squared_dist_point_to_itself_is_zero": {"atol": 1e-6},
            "geodesic_bvp_reverse": {"atol": 1e-4},
        },
    )
    # NB: points are sampled in a single grove
    testing_data.data_generator = RandomGroveDataGenerator(
        space, space.random_point().topology
    )


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

    space = WaldSpace(n_labels=random.randint(4, 5))

    request.cls.geodesic_solver = GeodesicSolver(space, **kwargs)
    request.cls.testing_data.space = space


@pytest.mark.usefixtures("geodesic_solvers")
class TestWaldGeodesicSolver(
    WaldGeodesicSolverTestCase, metaclass=DataBasedParametrizer
):
    testing_data = WaldGeodesicSolverTestData(
        point_counts=[1],
        method_defaults={"discrete_geodesic_reverse": {"atol": 1e-4}},
    )
