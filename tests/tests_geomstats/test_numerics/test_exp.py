import random

import pytest

from geomstats.geometry.invariant_metric import (
    InvariantMetric,
    InvariantMetricMatrixExpODESolver,
)
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.numerics.geodesic import ExpODESolver
from geomstats.numerics.ivp import GSIVPIntegrator, ScipySolveIVP
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.numerics.geodesic import (
    ExpSolverAgainstMetricTestCase,
    ExpSolverComparisonTestCase,
    ExpSolverTestCase,
)

from .data.geodesic import (
    ExpSolverAgainstMetricTestData,
    ExpSolverComparisonTestData,
    ExpSolverTestData,
)


def _create_params():
    params = []

    for space in (PoincareBall(random.randint(2, 3)),):
        for integrator in (
            GSIVPIntegrator(n_steps=20, step_type="rk4"),
            ScipySolveIVP(rtol=1e-8),
        ):
            solver = ExpODESolver(space, integrator=integrator)
            params.append((space, solver))

    return params


@pytest.fixture(
    scope="class",
    params=_create_params(),
)
def spaces(request):
    request.cls.space, request.cls.exp_solver = request.param


@pytest.mark.usefixtures("spaces")
class TestExpODESolverAgainstMetric(
    ExpSolverAgainstMetricTestCase, metaclass=DataBasedParametrizer
):
    """Test ExpODESolver with different integrators against closed-form implementations.

    NB: we lack closed-form solutions for matrix spaces. This justifies the creation
    of e.g. `TestExpODESolverMatrixComparison`.
    """

    testing_data = ExpSolverAgainstMetricTestData()


class TestExpODESolverMatrixComparison(
    ExpSolverComparisonTestCase, metaclass=DataBasedParametrizer
):
    """Test ExpODESolver solver for matrix spaces with different integrators.

    NB: `geodesic_ivp` is not implemented with `GSIVPIntegrator`.
    """

    space = SpecialOrthogonal(random.randint(2, 3), equip=False).equip_with_metric(
        InvariantMetric, left=True
    )
    space.metric.log_solver = None
    space.metric.exp_solver = None

    exp_solver = InvariantMetricMatrixExpODESolver(
        space,
        integrator=GSIVPIntegrator(n_steps=15, step_type="rk4"),
    )
    cmp_exp_solver = InvariantMetricMatrixExpODESolver(
        space, integrator=ScipySolveIVP(rtol=1e-8, point_ndim=2)
    )

    testing_data = ExpSolverComparisonTestData()


class TestExpODESolverMatrix(ExpSolverTestCase, metaclass=DataBasedParametrizer):
    """Test ExpODESolver with ScipySolveIVP for matrix points.

    Main goal is to test if `geodesic_ivp` runs, since it is not covered
    by any of the other tests in this file.
    """

    space = SpecialOrthogonal(random.randint(2, 3), equip=False).equip_with_metric(
        InvariantMetric, left=True
    )
    space.metric.log_solver = None
    space.metric.exp_solver = None

    exp_solver = InvariantMetricMatrixExpODESolver(
        space,
        integrator=ScipySolveIVP(
            rtol=1e-8,
            point_ndim=2,
        ),
    )

    testing_data = ExpSolverTestData()
