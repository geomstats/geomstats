import random

import pytest

from geomstats.geometry.discrete_curves import DiscreteCurves, ElasticMetric
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.stiefel import Stiefel
from geomstats.learning.frechet_mean import (
    BatchGradientDescent,
    FrechetMean,
    GradientDescent,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.frechet_mean import (
    BatchGradientDescentTestCase,
    CircularMeanTestCase,
    ElasticMeanTestCase,
    FrechetMeanTestCase,
)

from .data.frechet_mean import (
    BatchGradientDescentTestData,
    CircularMeanTestData,
    FrechetMeanTestData,
)

# TODO: add variance tests


@pytest.fixture(
    scope="class",
    params=[
        (Hypersphere(dim=random.randint(3, 4)), "default"),
        (Hypersphere(dim=random.randint(3, 4)), "adaptive"),
        (SpecialOrthogonal(n=3, point_type="vector"), "default"),
        (SpecialOrthogonal(n=3, point_type="vector"), "adaptive"),
        (SpecialOrthogonal(n=3, point_type="matrix"), "default"),
        (SpecialOrthogonal(n=3, point_type="matrix"), "adaptive"),
        (SPDMatrices(3), "default"),
        # (Stiefel(3, 2), "default"),  # TODO: create own test
        (Hyperboloid(dim=3), "default"),
    ],
)
def estimators(request):
    space, method = request.param
    request.cls.estimator = FrechetMean(space, method=method)


@pytest.mark.usefixtures("estimators")
class TestFrechetMean(FrechetMeanTestCase, metaclass=DataBasedParametrizer):
    testing_data = FrechetMeanTestData()


@pytest.fixture(
    scope="class",
    params=[
        Euclidean(dim=random.randint(2, 4)),
        Minkowski(dim=random.randint(2, 4)),
        Matrices(m=random.randint(2, 4), n=random.randint(2, 4)),
    ],
)
def linear_mean_estimators(request):
    space = request.param
    request.cls.estimator = FrechetMean(space)


@pytest.mark.usefixtures("linear_mean_estimators")
class TestLinearMean(FrechetMeanTestCase, metaclass=DataBasedParametrizer):
    testing_data = FrechetMeanTestData()


@pytest.fixture(
    scope="class",
    params=[
        DiscreteCurves(Euclidean(dim=2, equip=True), equip=False).equip_with_metric(
            ElasticMetric, a=1, b=1
        ),
        DiscreteCurves(Euclidean(dim=2, equip=True)),
    ],
)
def elastic_mean_estimators(request):
    space = request.param
    request.cls.estimator = FrechetMean(space)


@pytest.mark.usefixtures("elastic_mean_estimators")
class TestElasticMean(ElasticMeanTestCase, metaclass=DataBasedParametrizer):
    testing_data = FrechetMeanTestData()


class TestCircularMean(CircularMeanTestCase, metaclass=DataBasedParametrizer):
    space = Hypersphere(dim=1)
    estimator = FrechetMean(space)

    testing_data = CircularMeanTestData()


@pytest.fixture(
    scope="class",
    params=[
        Euclidean(dim=random.randint(2, 4)),
        Matrices(m=random.randint(2, 4), n=random.randint(2, 4)),
    ],
)
def batch_gradient_descent_estimators(request):
    request.cls.space = request.param
    request.cls.batch_optimizer = BatchGradientDescent()
    request.cls.optimizer = GradientDescent()


@pytest.mark.usefixtures("batch_gradient_descent_estimators")
class TestBatchGradientDescent(
    BatchGradientDescentTestCase, metaclass=DataBasedParametrizer
):
    testing_data = BatchGradientDescentTestData()
