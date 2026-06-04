import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.discrete_curves import (
    DiscreteCurvesStartingAtOrigin,
    ElasticMetric,
)
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import (
    BatchGradientDescent,
    FrechetMean,
    GradientDescent,
)
from geomstats.metric_geometry.bhv_space import TreeSpace
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
)
from geomstats.test_cases.learning.frechet_mean import (
    BatchGradientDescentTestCase,
    CircularMeanTestCase,
    ElasticMeanTestCase,
    FrechetMeanTestCase,
    PointSetFrechetMeanTestCase,
    VarianceTestCase,
)

from .data.frechet_mean import (
    BatchGradientDescentTestData,
    CircularMeanTestData,
    FrechetMeanSOCoincideTestData,
    FrechetMeanTestData,
    LinearMeanEuclideaTestData,
    PointSetFrechetMeanTestData,
    SturmMeanTestData,
    VarianceEuclideanTestData,
    VarianceTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        (Hypersphere(dim=random.randint(3, 4)), "default", None),
        (Hypersphere(dim=random.randint(3, 4)), "adaptive", None),
        (SPDMatrices(random.randint(3, 4)), "default", None),
        (SPDMatrices(random.randint(3, 4)), "adaptive", None),
    ],
)
def estimators(request):
    space, method, kwargs = request.param
    kwargs = kwargs or {}
    request.cls.estimator = FrechetMean(space, method=method, **kwargs)


@pytest.mark.usefixtures("estimators")
class TestFrechetMean(FrechetMeanTestCase, metaclass=DataBasedParametrizer):
    testing_data = FrechetMeanTestData()


@pytest.fixture(
    scope="class",
    params=[
        (PoincareBall(dim=random.randint(3, 4)), dict(max_iter=100)),
        (
            PoincareBall(dim=random.randint(3, 4)),
            dict(sample_method="stochastic", max_iter=500),
        ),
    ],
)
def estimators_for_sturm(request):
    space, kwargs = request.param
    kwargs = kwargs or {}
    request.cls.estimator = FrechetMean(space, method="sturms", **kwargs)


@pytest.mark.usefixtures("estimators_for_sturm")
class TestSturmsMean(FrechetMeanTestCase, metaclass=DataBasedParametrizer):
    testing_data = SturmMeanTestData()


@pytest.fixture(
    scope="class",
    params=[
        (TreeSpace(n_labels=5), "cyclic"),
        (TreeSpace(n_labels=5), "stochastic"),
    ]
    if not gs.__name__.endswith("pytorch")
    else [],
)
def pointset_estimators(request):
    space, sample_method = request.param

    request.cls.estimator = FrechetMean(
        space, method="sturms", sample_method=sample_method
    )


@pytest.mark.usefixtures("pointset_estimators")
class TestPointSetFrechetMean(
    PointSetFrechetMeanTestCase, metaclass=DataBasedParametrizer
):
    testing_data = PointSetFrechetMeanTestData()


class TestFrechetMeanSOCoincide(BaseEstimatorTestCase, metaclass=DataBasedParametrizer):
    estimator = FrechetMean(SpecialOrthogonal(n=3, point_type="matrix"))
    other_estimator = FrechetMean(SpecialOrthogonal(n=3, point_type="vector"))

    testing_data = FrechetMeanSOCoincideTestData()

    @pytest.mark.random
    def test_estimate_coincide(self, n_samples, atol):
        mat_space = self.estimator.space
        X = mat_space.random_point(n_samples)

        res = self.estimator.fit(X).estimate_

        X_vec = mat_space.rotation_vector_from_matrix(X)
        res_vec = self.other_estimator.fit(X_vec).estimate_
        res_ = mat_space.matrix_from_rotation_vector(res_vec)

        self.assertAllClose(res, res_, atol=atol)


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


@pytest.mark.smoke
class TestLinearMeanEuclidean(FrechetMeanTestCase, metaclass=DataBasedParametrizer):
    estimator = FrechetMean(Euclidean(dim=3))
    testing_data = LinearMeanEuclideaTestData()
    atol = 1e-6


@pytest.fixture(
    scope="class",
    params=[
        DiscreteCurvesStartingAtOrigin(ambient_dim=2, equip=False).equip_with_metric(
            ElasticMetric,
            a=1,
        ),
        DiscreteCurvesStartingAtOrigin(ambient_dim=2, equip=True),
    ],
)
def elastic_mean_estimators(request):
    space = request.param
    request.cls.estimator = FrechetMean(space)
    request.cls.atol = 1e-6


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
        Hypersphere(dim=random.randint(3, 4)),
        Hyperboloid(dim=3),
    ],
)
def variance_estimators(request):
    request.cls.space = request.param


@pytest.mark.usefixtures("variance_estimators")
class TestVariance(VarianceTestCase, metaclass=DataBasedParametrizer):
    testing_data = VarianceTestData()


@pytest.mark.smoke
class TestVarianceEuclidean(VarianceTestCase, metaclass=DataBasedParametrizer):
    space = Euclidean(dim=2)
    testing_data = VarianceEuclideanTestData()


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
    request.cls.atol = 1e-6


@pytest.mark.usefixtures("batch_gradient_descent_estimators")
class TestBatchGradientDescent(
    BatchGradientDescentTestCase, metaclass=DataBasedParametrizer
):
    testing_data = BatchGradientDescentTestData()
