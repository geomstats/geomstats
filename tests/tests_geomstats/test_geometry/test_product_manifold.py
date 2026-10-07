import pytest

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.siegel import Siegel
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
