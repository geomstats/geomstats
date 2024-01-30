import random

import pytest

from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.numerics.geodesic import ExpODESolver
from geomstats.numerics.ivp import GSIVPIntegrator, ScipySolveIVP
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.numerics.geodesic import (
    ExpSolverAgainstMetricTestCase,
    ExpSolverComparisonTestCase,
    ExpSolverTestCase,
    ExpSolverTypeCheck,
)

from .data.geodesic import (
    ExpSolverAgainstMetricTestData,
    ExpSolverComparisonTestData,
    ExpSolverTestData,
    ExpSolverTypeCheckTestData,
)


class _MatrixLieGroupRandomDataGenerator(RandomDataGenerator):
    def random_tangent_vec(self, base_point):
        """Random tangent vec.

        Applies tangent translation map to the generated tangent vector due
        to the way the geodesic equation is implemented.

        NB: it works properly in the context of these tests. Do not use outside.
        """
        tangent_vec = self.space.random_tangent_vec(base_point) / self.amplitude

        left_angular_vel = self.space.tangent_translation_map(
            base_point, left=self.space.metric.left, inverse=True
        )(tangent_vec)
        return left_angular_vel


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

    exp_solver = ExpODESolver(
        integrator=GSIVPIntegrator(n_steps=15, step_type="rk4"),
    )
    cmp_exp_solver = ExpODESolver(integrator=ScipySolveIVP(rtol=1e-8))

    data_generator = _MatrixLieGroupRandomDataGenerator(space)

    testing_data = ExpSolverComparisonTestData()


class TestExpODESolverMatrix(ExpSolverTestCase, metaclass=DataBasedParametrizer):
    """Test ExpODESolver with ScipySolveIVP for matrix points.

    Main goal is to test `geodesic_ivp` vectorization since it is not covered
    by any of the other tests.
    """

    space = SpecialOrthogonal(random.randint(2, 3), equip=False).equip_with_metric(
        InvariantMetric, left=True
    )
    space.metric.log_solver = None
    space.metric.exp_solver = None

    exp_solver = ExpODESolver(integrator=ScipySolveIVP(rtol=1e-8))

    data_generator = _MatrixLieGroupRandomDataGenerator(space)

    testing_data = ExpSolverTestData()


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
