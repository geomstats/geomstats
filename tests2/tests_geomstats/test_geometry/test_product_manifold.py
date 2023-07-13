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
from tests2.tests_geomstats.test_geometry.data.product_manifold import (
    ProductManifoldTestData,
    ProductRiemannianMetricTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        ((Hypersphere(dim=3, equip=False), Hyperboloid(dim=3, equip=False)), "matrix"),
        ((Hypersphere(dim=3, equip=False), Hyperboloid(dim=3, equip=False)), "vector"),
        ((Hypersphere(dim=3, equip=False), Hyperboloid(dim=4, equip=False)), "vector"),
        ((Hypersphere(dim=1, equip=False), Euclidean(dim=1, equip=False)), "vector"),
        (
            (SpecialOrthogonal(n=2, equip=False), SpecialOrthogonal(n=3, equip=False)),
            "vector",
        ),
        (
            (SpecialOrthogonal(n=2, equip=False), Euclidean(dim=3, equip=False)),
            "vector",
        ),
        (
            (
                Euclidean(dim=2, equip=False),
                Euclidean(dim=1, equip=False),
                Euclidean(dim=4, equip=False),
            ),
            "vector",
        ),
        (
            (Siegel(2, equip=False), Siegel(2, equip=False), Siegel(2, equip=False)),
            "other",
        ),
    ],
)
def spaces(request):
    factors, default_point_type = request.param
    request.cls.space = ProductManifold(
        factors=factors, default_point_type=default_point_type, equip=False
    )


@pytest.mark.usefixtures("spaces")
class TestProductManifold(ProductManifoldTestCase, metaclass=DataBasedParametrizer):
    testing_data = ProductManifoldTestData()


@pytest.fixture(
    scope="class",
    params=[
        ((Hypersphere(dim=3), Hyperboloid(dim=3)), "matrix"),
        ((Hypersphere(dim=3), Hyperboloid(dim=3)), "vector"),
        ((Hypersphere(dim=3), Hyperboloid(dim=4)), "vector"),
        ((Hypersphere(dim=1), Euclidean(dim=1)), "vector"),
        (
            (SpecialOrthogonal(n=2), SpecialOrthogonal(n=3)),
            "vector",
        ),
        (
            (SpecialOrthogonal(n=2), Euclidean(dim=3)),
            "vector",
        ),
        (
            (
                Euclidean(dim=2),
                Euclidean(dim=1),
                Euclidean(dim=4),
            ),
            "vector",
        ),
        (
            (Siegel(2), Siegel(2), Siegel(2)),
            "other",
        ),
    ],
)
def equipped_spaces(request):
    factors, default_point_type = request.param
    request.cls.space = ProductManifold(
        factors=factors,
        default_point_type=default_point_type,
        equip=True,
    )


@pytest.mark.usefixtures("equipped_spaces")
class TestProductRiemannianMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ProductRiemannianMetricTestData()
