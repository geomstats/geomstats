import random

import pytest

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.geometric_median import GeometricMedian
from geomstats.learning.riemannian_robust_m_estimator import (
    RiemannianAutoGradientDescent,
    RiemannianRobustMestimator,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import autograd_and_torch_only, np_only
from geomstats.test_cases.learning._base import BaseEstimatorTestCase
from geomstats.test_cases.learning.riemannian_robust_m_estimator import (
    AutoGradientDescentTestCase,
    DiffStartingPointSameResultTestCase,
    ExtremeHuberMeanCTestCase,
    SameMestimatorFunctionGivenByCustomAndExplicitTestCase,
    VarianceTestCase,
)

from .data.riemannian_robust_m_estimator import (
    AutoGradientDescentTestData,
    AutoGradientNotImplementedOnNumpyBackendTestData,
    DiffStartingPointSameResultTestData,
    ExtremeHuberMeanCTestData,
    RobustMestimatorSOCoincideTestData,
    SameMestimatorFunctionGivenByCustomAndExplicitTestData,
    VarianceEuclideanTestData,
    VarianceTestData,
)


class TestRobustMestimatorSOCoincide(BaseEstimatorTestCase, metaclass=DataBasedParametrizer):
    """Test SO matrix/vector coincidence."""

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
        """Test SO matrix/vector coincidence."""
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
    """Test huber limiting data inputs."""

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
class TestHuberMeanExtremeC(ExtremeHuberMeanCTestCase, metaclass=DataBasedParametrizer):
    """Test huber limiting."""

    testing_data = ExtremeHuberMeanCTestData()


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
    """Test autograd quality inputs."""

    space, m_estimator = request.param
    request.cls.estimator = RiemannianRobustMestimator(space, method='autograd', m_estimator=m_estimator, critical_value=1)


@autograd_and_torch_only
@pytest.mark.usefixtures("estimators_autograd")
class TestAutoGradientDescent(AutoGradientDescentTestCase, metaclass=DataBasedParametrizer):
    """Test autograd quality."""
    
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
    """Test Variance quality."""

    request.cls.space = request.param


@pytest.mark.usefixtures("variance_estimators")
class TestVariance(VarianceTestCase, metaclass=DataBasedParametrizer):
    """Test Variance quality."""

    testing_data = VarianceTestData()


@pytest.mark.smoke
class TestVarianceEuclidean(VarianceTestCase, metaclass=DataBasedParametrizer):
    """Test Euclidean Variance quality."""

    space = Euclidean(dim=2)
    testing_data = VarianceEuclideanTestData()


@pytest.fixture(
    scope="class",
    params=[
        (Hypersphere(dim=3),'default'),
        (Hypersphere(dim=3),'pseudo_huber'),
        (Hypersphere(dim=3),'cauchy'),
        (Hypersphere(dim=3),'biweight'),
        (Hypersphere(dim=3),'fair'),
        (Hypersphere(dim=3),'hampel'),
        (Hypersphere(dim=3),'welsch'),
        (Hypersphere(dim=3),'logistic'),
        (Hypersphere(dim=3),'lorentzian'),
        (Hypersphere(dim=3),'correntropy'),
        (PoincareBall(dim=3),'default'),
        (PoincareBall(dim=3),'pseudo_huber'),
        (PoincareBall(dim=3),'cauchy'),
        (PoincareBall(dim=3),'biweight'),
        (PoincareBall(dim=3),'fair'),
        (PoincareBall(dim=3),'hampel'),
        (PoincareBall(dim=3),'welsch'),
        (PoincareBall(dim=3),'logistic'),
        (PoincareBall(dim=3),'lorentzian'),
        (PoincareBall(dim=3),'correntropy'),
    ],
)
def estimators_starting_point(request):
    """Test starting point invariance inputs."""

    request.cls.space, m_estimator = request.param

    cutoff = 3 if m_estimator == 'biweight' else 1.5
    
    request.cls.estimator = RiemannianRobustMestimator(
        request.cls.space, m_estimator=m_estimator, method='default', init_point_method='mean-projection', critical_value=cutoff)
    request.cls.estimator_md = RiemannianRobustMestimator(
        request.cls.space, m_estimator=m_estimator, method='default', init_point_method='midpoint', critical_value=cutoff)
    request.cls.estimator_f = RiemannianRobustMestimator(
        request.cls.space, m_estimator=m_estimator, method='default', init_point_method='first', critical_value=cutoff)
    
    step_size = 5 if m_estimator == 'biweight' else 0.2
        
    request.cls.estimator.set(init_step_size=step_size, max_iter=2048, epsilon=1e-7, verbose=True)
    request.cls.estimator_md.set(init_step_size=step_size, max_iter=2048, epsilon=1e-7, verbose=True)
    request.cls.estimator_f.set(init_step_size=step_size, max_iter=2048, epsilon=1e-7, verbose=True)


@pytest.mark.usefixtures("estimators_starting_point")
class TestDiffStartingPointSameResult(DiffStartingPointSameResultTestCase, metaclass=DataBasedParametrizer):
    """Test starting point invariance."""

    testing_data = DiffStartingPointSameResultTestData()
    

@np_only
class TestAutoGradientNotImplementedOnNumpyBackend(BaseEstimatorTestCase, metaclass=DataBasedParametrizer):
    """Test autograd not working on numpy."""

    space = Hypersphere(dim=3)
    estimator = RiemannianRobustMestimator(
        space, 
        critical_value=1,
        m_estimator='huber',
        method='adaptive',
    )
    
    testing_data = AutoGradientNotImplementedOnNumpyBackendTestData()

    def test_auto_gradient_not_implemented_on_numpy_backend(self):
        """Test autograd not working on numpy."""
        with pytest.raises(NotImplementedError):
            RiemannianAutoGradientDescent(self.space)


@pytest.fixture(
    scope="class",
    params=[
        (Hypersphere(dim=3)),
        (PoincareBall(dim=3)),
        (SPDMatrices(n=3)),
        (SpecialOrthogonal(n=3, point_type="matrix")),
        (SpecialEuclidean(n=3)),
    ],
)
def estimators_custom_and_explicit(request):
    """Test custom function working inputs"""

    request.cls.space = request.param
    
    request.cls.estimator = RiemannianRobustMestimator(
        request.cls.space, m_estimator='cauchy', method='default', init_point_method='mean-projection', critical_value=1)
    request.cls.estimator.set(init_step_size=1, max_iter=4096, epsilon=1e-7, verbose=True)
    request.cls.estimator_custom = RiemannianRobustMestimator(
        request.cls.space, m_estimator='default', method='default', init_point_method='mean-projection', critical_value=1)
    request.cls.estimator_custom.set(init_step_size=1, max_iter=4096, epsilon=1e-7, verbose=True)
      

@pytest.mark.usefixtures("estimators_custom_and_explicit")
class TestSameMestimatorFunctionGivenByCustomAndExplicit(SameMestimatorFunctionGivenByCustomAndExplicitTestCase, metaclass=DataBasedParametrizer):
    """Test custom function working"""

    testing_data = SameMestimatorFunctionGivenByCustomAndExplicitTestData()