import pytest

import geomstats.backend as gs
from geomstats.numerics.optimization import (
    NewtonMethod,
    ScipyMinimize,
    ScipyRoot,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.numerics.optimization import (
    OptimizerTestCase,
    RootFinderTestCase,
)

from .data.optimization import (
    OptimizationHessSmokeTestData,
    OptimizationJacSmokeTestData,
    OptimizationSmokeTestData,
    RootFindingJacSmokeTestData,
    RootFindingSmokeTestData,
)

TORCH_OPTIMIZERS = []

try:
    from geomstats.numerics.optimization import (
        TorchAdam,
        TorchLBFGS,
        TorchRMSprop,
        TorchSGD,
    )
except ImportError:
    pass


try:
    from geomstats.numerics.optimization import TorchminMinimize
except ImportError:
    pass


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
    )
    + (
        [
            TorchLBFGS(),
            TorchSGD(lr=1e-2),
            TorchRMSprop(),
            TorchAdam(lr=1e-2),
            TorchminMinimize(),
        ]
        if gs.__name__.endswith("pytorch")
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


@pytest.fixture(
    scope="class",
    params=[
        ScipyRoot(),
    ]
    + ([ScipyRoot(autodiff_jac=True)] if gs.has_autodiff() else []),
)
def root_finders(request):
    request.cls.root_finder = request.param


@pytest.mark.smoke
@pytest.mark.usefixtures("root_finders")
class TestRootFinder(RootFinderTestCase, metaclass=DataBasedParametrizer):
    testing_data = RootFindingSmokeTestData()


@pytest.fixture(
    scope="class",
    params=[
        ScipyRoot(),
        NewtonMethod(),
    ],
)
def root_finders_with_jac(request):
    request.cls.root_finder = request.param


@pytest.mark.smoke
@pytest.mark.usefixtures("root_finders_with_jac")
class TestRootFinderJac(RootFinderTestCase, metaclass=DataBasedParametrizer):
    testing_data = RootFindingJacSmokeTestData()
