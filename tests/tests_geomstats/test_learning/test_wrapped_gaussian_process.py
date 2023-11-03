import pytest
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.wrapped_gaussian_process import WrappedGaussianProcess
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.wrapped_gaussian_process import (
    WrappedGaussianProcessTestCase,
)

from .data.wrapped_gaussian_process import WrappedGaussianProcessTestData


def _get_params():
    space = Hypersphere(dim=2)

    intercept_sphere_true = gs.array([0.0, -1.0, 0.0])
    coef_sphere_true = gs.array([1.0, 0.0, 0.5])

    prior = lambda x: space.metric.exp(
        x * coef_sphere_true,
        base_point=intercept_sphere_true,
    )

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))

    return space, prior, kernel


@pytest.fixture(
    scope="class",
    params=[
        _get_params(),
    ],
)
def estimators(request):
    space, prior, kernel = request.param
    request.cls.estimator = WrappedGaussianProcess(space, prior).set(kernel=kernel)


@pytest.mark.usefixtures("estimators")
class TestWrappedGaussianProcess(
    WrappedGaussianProcessTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = WrappedGaussianProcessTestData()
