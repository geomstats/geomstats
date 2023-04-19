import pytest

from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.numerics.geodesic_solvers import ExpIVPSolver
from geomstats.numerics.ivp_solvers import GSIntegrator, ScipySolveIVP
from geomstats.test.numerics.geodesic_solvers import ExpSolverComparisonTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.exp_solvers_data import ExpIVPSolverComparisonTestData

# TODO: test if output is in the right backend


def _create_params():
    # TODO: do this more in a fixture like behavior?
    params = []

    space = PoincareBall(2)
    for integrator in (
        GSIntegrator(n_steps=20, step_type="rk4"),
        ScipySolveIVP(rtol=1e-8),
    ):
        solver = ExpIVPSolver(integrator=integrator)
        params.append((space, solver))

    return params


@pytest.fixture(
    scope="class",
    params=_create_params(),
)
def spaces(request):
    request.cls.space, request.cls.exp_solver = request.param


@pytest.mark.usefixtures("spaces")
class TestExpIVPSolverComparison(
    ExpSolverComparisonTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ExpIVPSolverComparisonTestData()
