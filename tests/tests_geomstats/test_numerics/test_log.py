import random

import pytest

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
    PathStraightening,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import autodiff_backend, autodiff_only
from geomstats.test_cases.numerics.geodesic import (
    LogSolverAgainstMetricTestCase,
    LogSolverTestCase,
)

from .data.log import (
    LogODESolverMatrixTestData,
    LogSolverAgainstClosedFormTestData,
    PathStraighteningAgainstClosedFormTestData,
)

ALLOWS_AUTODIFF = autodiff_backend()


def _create_params_autodiff():
    params = []
    if not ALLOWS_AUTODIFF:
        return params

    spaces_1d = [PoincareBall(random.randint(2, 3))]

    for space in spaces_1d:
        for solver in (LogODESolver(n_nodes=10, use_jac=False),):
            params.append((space, solver))

    spaces_2d = [SPDMatrices(random.randint(2, 3))]

    for space in spaces_1d + spaces_2d:
        for solver in (
            LogShootingSolver(flatten=True),
            LogShootingSolver(flatten=False),
        ):
            params.append((space, solver))

    return params


@pytest.fixture(
    scope="class",
    params=_create_params_autodiff(),
)
def spaces(request):
    request.cls.space, request.cls.log_solver = request.param


@pytest.mark.usefixtures("spaces")
class TestLogSolverAgainstClosedForm(
    LogSolverAgainstMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = LogSolverAgainstClosedFormTestData()


@autodiff_only
class TestPathStraighteningAgainstClosedForm(
    LogSolverAgainstMetricTestCase, metaclass=DataBasedParametrizer
):
    """Test path-straightening against closed form.

    Not in above test for fine-grained control of tolerances.
    """

    _dim = random.randint(2, 3)
    space = PoincareBall(_dim)
    if ALLOWS_AUTODIFF:
        log_solver = PathStraightening()

    testing_data = PathStraighteningAgainstClosedFormTestData()


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
