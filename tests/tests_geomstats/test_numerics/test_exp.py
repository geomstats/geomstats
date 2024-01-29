import random

import pytest

from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.numerics.geodesic import ExpODESolver
from geomstats.numerics.ivp import GSIVPIntegrator, ScipySolveIVP
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.numerics.geodesic import (
    ExpSolverComparisonTestCase,
    ExpSolverTypeCheck,
)

from .data.exp import ExpODESolverComparisonTestData
from .data.geodesic import ExpSolverTypeCheckTestData


def _create_params():
    params = []

    for space in (PoincareBall(random.randint(2, 3)),):
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

    space = PoincareBall(random.randint(2, 3))
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
