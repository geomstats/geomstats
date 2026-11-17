import random

from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.invariant_metric import (
    InvariantMetric,
    InvariantMetricMatrixExpODESolver,
)
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.numerics.geodesic import ExpODESolver
from geomstats.numerics.ivp import GSIVPIntegrator, ScipySolveIVP
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


class TestExpODESolverGSIVPAgainstMetric(
    ExpSolverAgainstMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    """Test ExpODESolver with GSIVPIntegrator against closed-form implementations.

    NB: `geodesic_ivp` is not implemented with `GSIVPIntegrator`.
    """

    space = PoincareBall(random.randint(2, 3))
    exp_solver = ExpODESolver(
        space, integrator=GSIVPIntegrator(n_steps=20, step_type="rk4")
    )

    testing_data = ExpSolverAgainstMetricTestData(
        space, excluded_methods=("geodesic_ivp",)
    )


class TestExpODESolverScipyAgainstMetric(
    ExpSolverAgainstMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    """Test ExpODESolver with ScipySolveIVP against closed-form implementations.

    NB: we lack closed-form solutions for matrix spaces. This justifies the creation
    of e.g. `TestExpODESolverMatrixComparison`.
    """

    space = PoincareBall(random.randint(2, 3))
    exp_solver = ExpODESolver(space, integrator=ScipySolveIVP(rtol=1e-8))

    testing_data = ExpSolverAgainstMetricTestData(space)


class TestExpODESolverMatrixComparison(
    ExpSolverComparisonTestCase, metaclass=GeometricDataBasedParametrizer
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

    testing_data = ExpSolverComparisonTestData(
        space, excluded_methods=("geodesic_ivp",)
    )


class TestExpODESolverMatrix(
    ExpSolverTestCase, metaclass=GeometricDataBasedParametrizer
):
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

    testing_data = ExpSolverTestData(space, solver=exp_solver)
