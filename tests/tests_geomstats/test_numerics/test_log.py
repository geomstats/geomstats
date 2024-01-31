import random

import pytest

from geomstats.geometry.invariant_metric import (
    InvariantMetric,
    InvariantMetricMatrixLogODESolver,
)
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.numerics.geodesic import LogODESolver, LogShootingSolver
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.numerics.geodesic import (
    LogSolverAgainstMetricTestCase,
    LogSolverTestCase,
    LogSolverTypeCheckTestCase,
)

from .data.geodesic import LogSolverAgainstMetricTestData, LogSolverTypeCheckTestData
from .data.log import LogODESolverMatrixTestData


def _create_params():
    params = []

    spaces_1d = [PoincareBall(random.randint(2, 3))]
    spaces_2d = [SPDMatrices(random.randint(2, 3))]

    for space in spaces_1d + spaces_2d:
        for solver in (
            LogShootingSolver(flatten=True),
            LogShootingSolver(flatten=False),
        ):
            params.append((space, solver))

    for space in spaces_1d:
        for solver in (LogODESolver(n_nodes=10, use_jac=False),):
            params.append((space, solver))

    return params


@pytest.fixture(
    scope="class",
    params=_create_params(),
)
def spaces(request):
    request.cls.space, request.cls.log_solver = request.param


@pytest.mark.usefixtures("spaces")
class TestLogSolverAgainstMetric(
    LogSolverAgainstMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = LogSolverAgainstMetricTestData()


class TestLogODESolverMatrix(LogSolverTestCase, metaclass=DataBasedParametrizer):
    """Test log solvers for matrix spaces."""

    space = SpecialOrthogonal(random.randint(2, 3), equip=False).equip_with_metric(
        InvariantMetric, left=False
    )
    space.metric.log_solver = None
    log_solver = InvariantMetricMatrixLogODESolver(
        n_nodes=10,
        use_jac=False,
    )

    testing_data = LogODESolverMatrixTestData()


def _create_params_type_check():
    params = []

    space = PoincareBall(2)
    for solver in (
        LogShootingSolver(flatten=True),
        LogShootingSolver(flatten=False),
        LogODESolver(n_nodes=10, use_jac=False),
    ):
        params.append((space, solver))

    return params


@pytest.fixture(
    scope="class",
    params=_create_params_type_check(),
)
def spaces_for_type_checking(request):
    request.cls.space, request.cls.log_solver = request.param


@pytest.mark.usefixtures("spaces_for_type_checking")
class TestLogSolverTypeCheck(
    LogSolverTypeCheckTestCase, metaclass=DataBasedParametrizer
):
    testing_data = LogSolverTypeCheckTestData()
