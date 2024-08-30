import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.invariant_metric import (
    InvariantMetric,
    InvariantMetricMatrixLogODESolver,
)
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.numerics.geodesic import (
    LogODESolver,
    LogShootingSolver,
    MultiresPathStraightening,
    PathStraightening,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import autodiff_only
from geomstats.test_cases.numerics.geodesic import (
    LogSolverAgainstMetricTestCase,
    LogSolverTestCase,
)

from .data.log import (
    LogODESolverMatrixTestData,
    LogSolverAgainstClosedFormTestData,
    PathStraighteningAgainstClosedFormTestData,
)


def _create_params():
    params = []

    spaces_1d = [PoincareBall(random.randint(2, 3))]

    for space in spaces_1d:
        for solver in (LogODESolver(space, n_nodes=10, use_jac=False),):
            params.append((space, solver))

    spaces_2d = [SPDMatrices(random.randint(2, 3))]

    for space in spaces_1d + spaces_2d:
        for solver in (
            LogShootingSolver(space, flatten=True),
            LogShootingSolver(space, flatten=False),
        ):
            params.append((space, solver))

    return params


@pytest.fixture(
    scope="class",
    params=_create_params() if gs.has_autodiff() else (),
)
def spaces_with_log_solvers(request):
    request.cls.space, request.cls.log_solver = request.param


@pytest.mark.usefixtures("spaces_with_log_solvers")
class TestLogSolverAgainstClosedForm(
    LogSolverAgainstMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = LogSolverAgainstClosedFormTestData()


def _create_params_path_straightening():
    params = []

    _dim = random.randint(2, 3)
    space = PoincareBall(_dim)

    sym_list = [False] if gs.__name__.endswith("autograd") else [True, False]

    for sym in sym_list:
        params.append(
            (
                space,
                PathStraightening(space, n_nodes=100, symmetric=sym),
            ),
        )

    for sym in sym_list:
        params.append(
            (
                space,
                MultiresPathStraightening(
                    space, n_levels=random.randint(4, 5), symmetric=sym
                ),
            )
        )

    return params


@pytest.fixture(
    scope="class",
    params=_create_params_path_straightening() if gs.has_autodiff() else (),
)
def spaces_with_path_straightening(request):
    request.cls.space, request.cls.log_solver = request.param


@autodiff_only
@pytest.mark.usefixtures("spaces_with_path_straightening")
class TestPathStraighteningAgainstClosedForm(
    LogSolverAgainstMetricTestCase, metaclass=DataBasedParametrizer
):
    """Test path-straightening against closed form.

    Not in above test for fine-grained control of tolerances.
    """

    testing_data = PathStraighteningAgainstClosedFormTestData()


class TestLogODESolverMatrix(LogSolverTestCase, metaclass=DataBasedParametrizer):
    """Test log solvers for matrix spaces."""

    space = SpecialOrthogonal(random.randint(2, 3), equip=False).equip_with_metric(
        InvariantMetric, left=False
    )
    space.metric.log_solver = None
    log_solver = InvariantMetricMatrixLogODESolver(
        space,
        n_nodes=10,
        use_jac=False,
    )

    testing_data = LogODESolverMatrixTestData()
