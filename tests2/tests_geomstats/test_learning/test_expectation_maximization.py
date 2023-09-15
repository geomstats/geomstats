import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.learning.expectation_maximization import (
    GaussianMixtureModel,
    RiemannianEM,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import autodiff_backend, autograd_and_torch_only
from geomstats.test_cases.learning.expectation_maximization import (
    GaussianMixtureModelTestCase,
    RiemannianEMTestCase,
)

from .data.expectation_maximization import (
    GaussianMixtureModelTestData,
    RiemannianEMTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        (PoincareBall(dim=2), "random"),
        (PoincareBall(dim=2), "kmeans"),
    ],
)
def estimators(request):
    space, initialisation_method = request.param

    request.cls.estimator = RiemannianEM(
        space,
        initialisation_method=initialisation_method,
        n_gaussians=random.randint(2, 4),
    )


@pytest.mark.usefixtures("estimators")
class TestRiemannianEM(RiemannianEMTestCase, metaclass=DataBasedParametrizer):
    testing_data = RiemannianEMTestData()


@autograd_and_torch_only
class TestRiemannianEMHypersphere(
    RiemannianEMTestCase, metaclass=DataBasedParametrizer
):
    space = Hypersphere(dim=2)
    if autodiff_backend():
        estimator = RiemannianEM(
            space,
            initialisation_method="random",
            n_gaussians=random.randint(2, 4),
        )

    testing_data = RiemannianEMTestData()


@pytest.mark.smoke
class TestGaussianMixtureModel(
    GaussianMixtureModelTestCase, metaclass=DataBasedParametrizer
):
    space = PoincareBall(dim=2)
    model = GaussianMixtureModel(
        space,
        zeta_lower_bound=5e-2,
        zeta_upper_bound=2.0,
        zeta_step=0.001,
        variances=gs.array([0.8, 1.2]),
    )

    testing_data = GaussianMixtureModelTestData()
