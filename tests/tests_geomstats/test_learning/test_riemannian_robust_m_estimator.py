import random
# import os
# os.environ["GEOMSTATS_BACKEND"] = "autograd"  ## TODO
# import geomstats.backend as gs
# print(gs.has_autodiff())
import pytest

from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.geometry.poincare_ball  import PoincareBall
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.spd_matrices import (
    SPDLogEuclideanMetric,
    SPDMatrices,
)
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.riemannian_robust_m_estimator import (
    RiemannianRobustMestimator,
    BaseGradientDescent,
    RiemannianAutoGradientDescent,
)
from geomstats.test.test_case import autograd_and_torch_only
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning._base import BaseEstimatorTestCase
from geomstats.test_cases.learning.riemannian_robust_m_estimator import (
    AutoGradientDescentTestCase,
    HuberMeanExtremeCTestCase,
    VarianceTestCase,
)

from .data.riemannian_robust_m_estimator import (
    AutoGradientDescentTestData,
    HuberMeanExtremeCTestData,
    RobustMestimatorSOCoincideTestData,
    VarianceEuclideanTestData,
    VarianceTestData,
)
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.geometric_median import GeometricMedian


class TestRobustMestimatorSOCoincide(BaseEstimatorTestCase, metaclass=DataBasedParametrizer):
    estimator = RiemannianRobustMestimator(
        SpecialOrthogonal(n=3, point_type="matrix"), 
        critical_value=1,
        m_estimator='huber',
        method='adaptive',
    )
    other_estimator = RiemannianRobustMestimator(
        SpecialOrthogonal(n=3, point_type="vector"),
        critical_value=1,
        m_estimator='huber',
        method='adaptive',
    )

    testing_data = RobustMestimatorSOCoincideTestData()

    @pytest.mark.random
    def test_estimate_coincide(self, n_samples, atol):
        mat_space = self.estimator.space
        X = mat_space.random_point(n_samples)

        res = self.estimator.fit(X).estimate_.x

        X_vec = mat_space.rotation_vector_from_matrix(X)
        res_vec = self.other_estimator.fit(X_vec).estimate_.x
        res_ = mat_space.matrix_from_rotation_vector(res_vec)

        self.assertAllClose(res, res_, atol=atol)


@pytest.fixture(
    scope="class",
    params=[
        (Hypersphere(dim=random.randint(3, 4))),
        (PoincareBall(dim=random.randint(2, 3))),
        (SPDMatrices(n=3)),
        (SpecialOrthogonal(n=3, point_type="matrix")),
        (SpecialEuclidean(n=random.randint(3, 4))),
    ],
)
def estimators_huber_extreme_c(request):
    space = request.param
    request.cls.estimator = RiemannianRobustMestimator(
        space, m_estimator='huber', method='default', init_point_method='mean-projection', critical_value=1e-10)
    request.cls.estimator.set(init_step_size=1e7*5, max_iter=4096, epsilon=1e-16, verbose=True)
    request.cls.estimator_inf = RiemannianRobustMestimator(
        space, m_estimator='huber', method='adaptive', init_point_method='mean-projection', critical_value=1e10)
    request.cls.estimator_inf.set(init_step_size=100, max_iter=4096, epsilon=1e-16)
    request.cls.estimator_frechet_mean = FrechetMean(space)
    request.cls.estimator_frechet_mean.set(max_iter=1024, epsilon=1e-16)
    request.cls.estimator_geometric_median = GeometricMedian(space,lr=1, max_iter=4096, epsilon=1e-16)
    

@pytest.mark.usefixtures("estimators_huber_extreme_c")
class TestHuberMeanExtremeC(HuberMeanExtremeCTestCase, metaclass=DataBasedParametrizer):
    testing_data = HuberMeanExtremeCTestData()


@pytest.fixture(
    scope="class",
    params=[
        (Hypersphere(dim=random.randint(3, 4)),'huber'),
        (PoincareBall(dim=3), "fair"),
        (PoincareHalfSpace(dim=random.randint(2, 3)), "biweight"),
        (SpecialOrthogonal(n=random.randint(2, 3)), "hampel"),
        (SpecialEuclidean(n=random.randint(3, 4)), "cauchy"),
    ],
)
def estimators_autograd(request):
    space, m_estimator = request.param
    request.cls.estimator = RiemannianRobustMestimator(space, method='autograd', m_estimator=m_estimator, critical_value=1)


@autograd_and_torch_only
@pytest.mark.usefixtures("estimators_autograd")
class TestAutoGradientDescent(AutoGradientDescentTestCase, metaclass=DataBasedParametrizer):
    testing_data = AutoGradientDescentTestData()


@pytest.fixture(
    scope="class",
    params=[
        Hypersphere(dim=random.randint(3, 4)),
        Hyperboloid(dim=3),
        SPDMatrices(n=3),
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