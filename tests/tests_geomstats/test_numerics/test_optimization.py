import pytest

import geomstats.backend as gs
from geomstats.numerics.optimizers import ScipyMinimize
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.numerics.optimization import (
    OptimizerTestCase,
)

from .data.optimization import (
    OptimizationHessSmokeTestData,
    OptimizationJacSmokeTestData,
    OptimizationSmokeTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        ScipyMinimize(method="BFGS"),
    ]
    + (
        [
            ScipyMinimize(method="BFGS", autodiff_jac=True),
            ScipyMinimize(method="Newton-CG", autodiff_jac=True),
            ScipyMinimize(method="Newton-CG", autodiff_jac=True, autodiff_hess=True),
        ]
        if gs.has_autodiff()
        else []
    ),
)
def optimizers(request):
    request.cls.optimizer = request.param


@pytest.mark.smoke
@pytest.mark.usefixtures("optimizers")
class TestOptimizers(OptimizerTestCase, metaclass=DataBasedParametrizer):
    testing_data = OptimizationSmokeTestData()


@pytest.fixture(
    scope="class",
    params=[
        ScipyMinimize(method="BFGS"),
        ScipyMinimize(method="Newton-CG"),
    ]
    + (
        [
            ScipyMinimize(method="Newton-CG", autodiff_hess=True),
        ]
        if gs.has_autodiff()
        else []
    ),
)
def optimizers_with_jac(request):
    request.cls.optimizer = request.param


@pytest.mark.smoke
@pytest.mark.usefixtures("optimizers_with_jac")
class TestOptimizersJac(OptimizerTestCase, metaclass=DataBasedParametrizer):
    testing_data = OptimizationJacSmokeTestData()


@pytest.fixture(
    scope="class",
    params=[
        ScipyMinimize(method="Newton-CG"),
    ],
)
def optimizers_with_hess(request):
    request.cls.optimizer = request.param


@pytest.mark.smoke
@pytest.mark.usefixtures("optimizers_with_hess")
class TestOptimizersHess(OptimizerTestCase, metaclass=DataBasedParametrizer):
    testing_data = OptimizationHessSmokeTestData()
