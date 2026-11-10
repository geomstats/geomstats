import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hermitian import Hermitian
from geomstats.geometry.hermitian_matrices import HermitianMatrices
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
from geomstats.test_cases.geometry.product_manifold import (
    ProductManifoldTestCase,
    ProductRiemannianMetricTestCase,
)
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.product_manifold import (
    ProductManifoldTestData,
    ProductRiemannianMetricTestData,
)
from .data.riemannian_metric import RiemannianMetricTestData

FACTORS_AND_POINT_NDIM = [
    (((Euclidean, (3,)), (Euclidean, (3,))), 2),
    (((Euclidean, (3,)), (Euclidean, (3,))), 1),
    (((Euclidean, (3,)), (Euclidean, (4,))), 1),
    (((Euclidean, (2,)), (Euclidean, (1,)), (Euclidean, (4,))), 1),
    (((Matrices, (2, 3)),) * 2, 3),
    (((Matrices, (2, 3)), (Matrices, (3, 3))), 1),
    (((Matrices, (2, 2)), (Euclidean, (3,))), 1),
    (((Hermitian, (2,)),) * 2, 2),
    (((Hermitian, (2,)), (Hermitian, (3,))), 1),
    (((HermitianMatrices, (2,)),) * 3, 3),
    (((Euclidean, (2,)), (Hermitian, (2,))), 1),
    (((Matrices, (2, 2)), (HermitianMatrices, (2,))), 3),
    (((Matrices, (2, 2)), (HermitianMatrices, (2,))), 1),
    (((Euclidean, (3,)), (Euclidean, (3,))), None),
    (((Euclidean, (3,)), (Euclidean, (4,))), None),
    (((Hypersphere, (2,)), (Euclidean, (3,))), 1),
]


def _product_manifold(factors_and_point_ndim, equip=True):
    factors, point_ndim = factors_and_point_ndim
    return ProductManifold(
        factors=[factor(*args, equip=equip) for factor, args in factors],
        point_ndim=point_ndim,
        equip=equip,
    )


@pytest.fixture(scope="class", params=FACTORS_AND_POINT_NDIM)
def spaces(request):
    space = request.cls.space = _product_manifold(request.param, equip=False)

    request.cls.testing_data.space = space


@pytest.mark.usefixtures("spaces")
class TestProductManifold(
    ProductManifoldTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = ProductManifoldTestData()


@pytest.fixture(scope="class", params=FACTORS_AND_POINT_NDIM)
def equipped_spaces(request):
    space = request.cls.space = _product_manifold(request.param)

    request.cls.testing_data.space = space


@pytest.mark.usefixtures("equipped_spaces")
class TestProductRiemannianMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = RiemannianMetricTestData()


class TestScaledEuclideanProductMetric(
    ProductRiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ProductManifold(
        factors=[
            Euclidean(2).equip_with_metric(ScalarProductMetric, scale=2.0),
            Euclidean(3).equip_with_metric(ScalarProductMetric, scale=3.0),
        ],
        point_ndim=1,
    )
    testing_data = ProductRiemannianMetricTestData(space)
