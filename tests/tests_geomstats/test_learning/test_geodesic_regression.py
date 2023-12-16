import random

import pytest

from geomstats.geometry.discrete_curves import DiscreteCurvesStartingAtOrigin
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.learning.geodesic_regression import GeodesicRegression
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import autograd_only
from geomstats.test_cases.learning.geodesic_regression import GeodesicRegressionTestCase

from .data.geodesic_regression import GeodesicRegressionTestData

# TODO: test loss decreases for `RiemannianGradientDescent`? bring callback
# TODO: add tests for initialization (keep it simple)


@pytest.fixture(
    scope="class",
    params=[
        (Euclidean(random.randint(3, 5)), "extrinsic"),
        (Euclidean(random.randint(3, 5)), "riemannian"),
        (Hypersphere(random.randint(3, 5)), "extrinsic"),
        (Hypersphere(random.randint(3, 5)), "riemannian"),
        (
            DiscreteCurvesStartingAtOrigin(
                ambient_dim=2,
                k_sampling_points=random.randint(5, 10),
            ),
            "extrinsic",
        ),
    ],
)
def estimators(request):
    space, method = request.param
    request.cls.estimator = GeodesicRegression(space, method=method)


@pytest.mark.slow
@pytest.mark.usefixtures("estimators")
class TestGeodesicRegression(
    GeodesicRegressionTestCase, metaclass=DataBasedParametrizer
):
    testing_data = GeodesicRegressionTestData()


@pytest.fixture(
    scope="class",
    params=[
        (SpecialEuclidean(n=2), "extrinsic"),
        (SpecialEuclidean(n=2), "riemannian"),
    ],
)
def estimators2(request):
    space, method = request.param
    request.cls.estimator = GeodesicRegression(space, method=method)


@autograd_only
@pytest.mark.slow
@pytest.mark.usefixtures("estimators2")
class TestGeodesicRegressionOnlyAutograd(
    GeodesicRegressionTestCase, metaclass=DataBasedParametrizer
):
    testing_data = GeodesicRegressionTestData()
