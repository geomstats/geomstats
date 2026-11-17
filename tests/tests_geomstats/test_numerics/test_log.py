import random

import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

import geomstats.backend as gs
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
    MultiresPathStraightening,
    PathStraightening,
    PolynomialPathLogSolver,
)
from geomstats.test.test_case import autodiff_only
from geomstats.test_cases.numerics.geodesic import (
    LogSolverAgainstMetricTestCase,
    LogSolverTestCase,
)

from .data.geodesic import LogSolverAgainstMetricTestData, LogSolverTestData


def _create_params_ode():
    space = PoincareBall(random.randint(2, 3))
    return [
        (space, LogODESolver(space, n_nodes=10, use_jac=use_jac))
        for use_jac in (False, True)
    ]


@pytest.fixture(scope="class", params=_create_params_ode())
def spaces_with_ode_solvers(request):
    space, request.cls.log_solver = request.param
    request.cls.space = space
    request.cls.testing_data.space = space


@pytest.mark.usefixtures("spaces_with_ode_solvers")
class TestLogODESolverAgainstClosedForm(
    LogSolverAgainstMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    """Test LogODESolver, with and without Jacobian, against closed form."""

    testing_data = LogSolverAgainstMetricTestData(
        method_defaults={
            "log": {"atol": 1e-3},
            "geodesic_bvp": {"atol": 1e-4},
        },
    )


def _create_params_shooting():
    params = []

    for space in (
        PoincareBall(random.randint(2, 3)),
        SPDMatrices(random.randint(2, 3)),
    ):
        for flatten in (True, False):
            params.append((space, LogShootingSolver(space, flatten=flatten)))

    return params


@pytest.fixture(
    scope="class",
    params=_create_params_shooting() if gs.has_autodiff() else (),
)
def spaces_with_shooting_solvers(request):
    space, request.cls.log_solver = request.param
    request.cls.space = space
    request.cls.testing_data.space = space


@autodiff_only
@pytest.mark.usefixtures("spaces_with_shooting_solvers")
class TestLogShootingSolverAgainstClosedForm(
    LogSolverAgainstMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    """Test shooting solvers against closed form.

    NB: `geodesic_bvp` is not implemented for shooting solvers.
    """

    testing_data = LogSolverAgainstMetricTestData(
        excluded_methods=("geodesic_bvp",),
        method_defaults={"log": {"atol": 1e-3}},
    )


def _create_params_path_based():
    params = []

    space = PoincareBall(random.randint(2, 3))

    sym_list = [False] if gs.__name__.endswith("autograd") else [True, False]

    for sym in sym_list:
        params.append(
            (
                space,
                PathStraightening(space, n_nodes=100, symmetric=sym),
            ),
        )

    for sym in sym_list:
        params.append(
            (
                space,
                MultiresPathStraightening(
                    space, n_levels=random.randint(4, 5), symmetric=sym
                ),
            )
        )

    params.append((space, PolynomialPathLogSolver(space)))

    return params


@pytest.fixture(
    scope="class",
    params=_create_params_path_based() if gs.has_autodiff() else (),
)
def spaces_with_path_based_solvers(request):
    space, request.cls.log_solver = request.param
    request.cls.space = space
    request.cls.testing_data.space = space


@autodiff_only
@pytest.mark.usefixtures("spaces_with_path_based_solvers")
class TestPathBasedLogSolverAgainstClosedForm(
    LogSolverAgainstMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    """Test path-based log solvers against closed form.

    Not in above test for fine-grained control of tolerances.
    """

    testing_data = LogSolverAgainstMetricTestData(
        method_defaults={
            "log": {"atol": 1e-1},
            "geodesic_bvp": {"atol": 1e-2},
        },
    )


class TestLogODESolverMatrix(
    LogSolverTestCase, metaclass=GeometricDataBasedParametrizer
):
    """Test log solvers for matrix spaces."""

    space = SpecialOrthogonal(random.randint(2, 3), equip=False).equip_with_metric(
        InvariantMetric, left=False
    )
    space.metric.log_solver = None
    log_solver = InvariantMetricMatrixLogODESolver(
        space,
        n_nodes=10,
        use_jac=False,
    )

    testing_data = LogSolverTestData(
        space,
        solver=log_solver,
        excluded_methods=("geodesic_bvp",),
        method_defaults={
            "geodesic_bvp_known_geod": {"atol": 1e-3},
            "log_after_exp": {"atol": 5e-4},
        },
    )
