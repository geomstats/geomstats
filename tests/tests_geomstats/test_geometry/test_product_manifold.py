import pytest

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.siegel import Siegel
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.product_manifold import ProductManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.product_manifold import (
    ProductManifoldTestData,
    ProductRiemannianMetricTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        ((Hypersphere(dim=3, equip=False), Hyperboloid(dim=3, equip=False)), 2),
        ((Hypersphere(dim=3, equip=False), Hyperboloid(dim=3, equip=False)), 1),
        ((Hypersphere(dim=3, equip=False), Hyperboloid(dim=4, equip=False)), 1),
        ((Hypersphere(dim=1, equip=False), Euclidean(dim=1, equip=False)), 1),
        (
            (SpecialOrthogonal(n=2, equip=False), SpecialOrthogonal(n=3, equip=False)),
            1,
        ),
        (
            (SpecialOrthogonal(n=2, equip=False), Euclidean(dim=3, equip=False)),
            1,
        ),
        (
            (
                Euclidean(dim=2, equip=False),
                Euclidean(dim=1, equip=False),
                Euclidean(dim=4, equip=False),
            ),
            1,
        ),
        (
            (Siegel(2, equip=False), Siegel(2, equip=False), Siegel(2, equip=False)),
            3,
        ),
    ],
)
def spaces(request):
    factors, point_ndim = request.param
    request.cls.space = ProductManifold(
        factors=factors, point_ndim=point_ndim, equip=False
    )


@pytest.mark.usefixtures("spaces")
class TestProductManifold(ProductManifoldTestCase, metaclass=DataBasedParametrizer):
    testing_data = ProductManifoldTestData()


@pytest.fixture(
    scope="class",
    params=[
        ((Hypersphere(dim=3), Hyperboloid(dim=3)), 2),
        ((Hypersphere(dim=3), Hyperboloid(dim=3)), 1),
        ((Hypersphere(dim=3), Hyperboloid(dim=4)), 1),
        ((Hypersphere(dim=1), Euclidean(dim=1)), 1),
        (
            (SpecialOrthogonal(n=2), SpecialOrthogonal(n=3)),
            1,
        ),
        (
            (SpecialOrthogonal(n=2), Euclidean(dim=3)),
            1,
        ),
        (
            (
                Euclidean(dim=2),
                Euclidean(dim=1),
                Euclidean(dim=4),
            ),
            1,
        ),
        (
            (Siegel(2), Siegel(2), Siegel(2)),
            3,
        ),
    ],
)
def equipped_spaces(request):
    factors, point_ndim = request.param
    request.cls.space = ProductManifold(
        factors=factors,
        point_ndim=point_ndim,
    )


@pytest.mark.usefixtures("equipped_spaces")
class TestProductRiemannianMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ProductRiemannianMetricTestData()


def test_mul_constructor():
    a = SpecialOrthogonal(n=3)
    b = SPDMatrices(n=4)
    product_1 = a * b
    assert isinstance(product_1, ProductManifold)

    product_2 = SpecialOrthogonal(n=2) * Euclidean(dim=2)
    assert isinstance(product_2, ProductManifold)

    product_3 = Euclidean(dim=1) * SPDMatrices(n=1)
    assert isinstance(product_3, ProductManifold)


def test_mul_constructor_vect():
    product = SpecialOrthogonal(n=3) * SPDMatrices(n=4)
    one_point = product.random_point()
    assert one_point.shape[0] == one_point.size

    mult_points = product.random_point(n_samples=5)
    assert mult_points.shape[0] == 5
    assert mult_points.shape[1] == one_point.size


def test_mul_invalid_factors():
    with pytest.raises(TypeError):
        _ = SpecialOrthogonal(3) * 5
