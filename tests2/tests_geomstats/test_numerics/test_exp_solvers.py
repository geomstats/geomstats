import pytest

from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.numerics.geodesic_solvers import ExpIVPSolver
from geomstats.numerics.ivp_solvers import GSIntegrator, ScipySolveIVP
from geomstats.test.numerics.geodesic_solvers import (
    ExpSolverComparisonTestCase,
    ExpSolverTypeCheck,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.exp_solvers_data import ExpIVPSolverComparisonTestData
from tests2.data.geodesic_solvers_data import ExpSolverTypeCheckTestData


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


def _create_params_type_check():
    params = []

    space = PoincareBall(2)
    for integrator in (
        GSIntegrator(n_steps=10, step_type="euler"),
        ScipySolveIVP(),
    ):
        solver = ExpIVPSolver(integrator=integrator)
        params.append((space, solver))

    return params


@pytest.fixture(
    scope="class",
    params=_create_params_type_check(),
)
def spaces_for_type_checking(request):
    request.cls.space, request.cls.exp_solver = request.param


@pytest.mark.usefixtures("spaces_for_type_checking")
class TestExpSolverTypeCheck(ExpSolverTypeCheck, metaclass=DataBasedParametrizer):
    testing_data = ExpSolverTypeCheckTestData()
