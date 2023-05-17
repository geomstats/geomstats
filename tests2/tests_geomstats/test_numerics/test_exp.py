import pytest

from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.numerics.geodesic import ExpODESolver
from geomstats.numerics.ivp import GSIVPIntegrator, ScipySolveIVP
from geomstats.test.numerics.geodesic_solvers import (
    ExpSolverComparisonTestCase,
    ExpSolverTypeCheck,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.exp_solvers_data import ExpODESolverComparisonTestData
from tests2.data.geodesic_solvers_data import ExpSolverTypeCheckTestData


def _create_params():
    # TODO: do this more in a fixture like behavior?
    params = []

    for space in (
        PoincareBall(2),
        PoincareBall(3),
    ):
        for integrator in (
            GSIVPIntegrator(n_steps=20, step_type="rk4"),
            ScipySolveIVP(rtol=1e-8),
        ):
            solver = ExpODESolver(integrator=integrator)
            params.append((space, solver))

    return params


@pytest.fixture(
    scope="class",
    params=_create_params(),
)
def spaces(request):
    request.cls.space, request.cls.exp_solver = request.param


@pytest.mark.usefixtures("spaces")
class TestExpODESolverComparison(
    ExpSolverComparisonTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ExpODESolverComparisonTestData()


def _create_params_type_check():
    params = []

    space = PoincareBall(2)
    for integrator in (
        GSIVPIntegrator(n_steps=10, step_type="euler"),
        ScipySolveIVP(),
    ):
        solver = ExpODESolver(integrator=integrator)
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
