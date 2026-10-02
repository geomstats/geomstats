import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.siegel import Siegel
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import assert_allclose
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


@pytest.mark.parametrize("point_ndim", [1, 2])
@pytest.mark.parametrize("path_arg", ["direction", "end_point"])
def test_parallel_transport_matches_factors(point_ndim, path_arg):
    """Product transport splits over factors and preserves tangent norms."""
    factors = (Hypersphere(dim=2), Hyperboloid(dim=2))
    space = ProductManifold(factors, point_ndim=point_ndim)
    base_points = [gs.array([1.0, 0.0, 0.0])] * 2
    tangent_vecs = [
        gs.array([[0.0, 0.2, 0.3], [0.0, -0.1, 0.4]]),
        gs.array([[0.0, 0.4, -0.1], [0.0, 0.2, 0.3]]),
    ]
    directions = [
        gs.array([[0.0, 0.3, -0.2], [0.0, -0.4, 0.2]]),
        gs.array([[0.0, -0.2, 0.1], [0.0, 0.1, 0.3]]),
    ]
    end_points = [
        factor.metric.exp(direction, base_point)
        for factor, direction, base_point in zip(factors, directions, base_points)
    ]
    path_args = directions if path_arg == "direction" else end_points
    expected = [
        factor.metric.parallel_transport(tangent_vec, base_point, **{path_arg: arg})
        for factor, tangent_vec, base_point, arg in zip(
            factors, tangent_vecs, base_points, path_args
        )
    ]
    base_point = space.embed_to_product(base_points)
    tangent_vec = space.embed_to_product(tangent_vecs)
    end_point = space.embed_to_product(end_points)

    transported = space.metric.parallel_transport(
        tangent_vec, base_point, **{path_arg: space.embed_to_product(path_args)}
    )

    assert transported.shape == tangent_vec.shape
    assert_allclose(transported, space.embed_to_product(expected))
    assert gs.all(space.is_tangent(transported, end_point))
    assert_allclose(
        space.metric.norm(transported, end_point),
        space.metric.norm(tangent_vec, base_point),
    )
