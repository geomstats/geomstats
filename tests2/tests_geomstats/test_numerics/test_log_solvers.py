import pytest

from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.numerics.geodesic_solvers import LogBVPSolver, LogShootingSolver
from geomstats.test.numerics.geodesic_solvers import LogSolverComparisonTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.geodesic_solvers_data import LogSolverComparisonTestData

# TODO: test if output is in the right backend


def _create_params():
    # TODO: do this more in a fixture like behavior?
    params = []

    space = PoincareBall(2)
    for solver in (
        LogShootingSolver(flatten=True),
        LogShootingSolver(flatten=False),
        LogBVPSolver(n_nodes=10),
    ):
        params.append((space, solver))

    return params


@pytest.fixture(
    scope="class",
    params=_create_params(),
)
def spaces(request):
    request.cls.space, request.cls.log_solver = request.param


@pytest.mark.usefixtures("spaces")
class TestLogSolverComparison(
    LogSolverComparisonTestCase, metaclass=DataBasedParametrizer
):
    testing_data = LogSolverComparisonTestData()
