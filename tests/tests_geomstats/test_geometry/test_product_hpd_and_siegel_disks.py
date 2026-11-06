import random

import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.hpd_matrices import HPDAffineMetric, HPDMatrices
from geomstats.geometry.product_hpd_and_siegel_disks import (
    ProductHPDMatricesAndSiegelDisks,
    ProductHPDMatricesAndSiegelDisksMetric,
)
from geomstats.geometry.siegel import Siegel
from geomstats.test_cases.geometry.product_manifold import ProductManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.product_manifold import ProductManifoldTestData
from .data.riemannian_metric import RiemannianMetricTestData


@pytest.mark.redundant
class TestProductHPDMatricesAndSiegelDisks(
    ProductManifoldTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ProductHPDMatricesAndSiegelDisks(
        n_manifolds=random.randint(2, 5), n=random.randint(2, 5), equip=False
    )
    testing_data = ProductManifoldTestData(space)


@pytest.mark.redundant
class TestProductHPDMatricesAndSiegelDisksMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ProductHPDMatricesAndSiegelDisks(
        n_manifolds=random.randint(2, 5), n=random.randint(2, 5)
    )
    testing_data = RiemannianMetricTestData(
        space,
        excluded_methods=(
            "parallel_transport_ivp",
            "parallel_transport_bvp",
            "parallel_transport_ivp_transported_is_tangent",
            "parallel_transport_bvp_transported_is_tangent",
            "parallel_transport_ivp_norm",
            "parallel_transport_bvp_norm",
        ),
        method_defaults={
            "dist_is_log_norm": {"atol": 1e-4},
            "geodesic_bvp_reverse": {"atol": 1e-4},
            "geodesic_ivp_belongs": {"atol": 1e-4},
            "exp_belongs": {"atol": 1e-4},
        },
    )


@pytest.mark.smoke
def test_product_hpd_matrices_and_siegel_disks_structure():
    """Check the factors, scales, shape and metric of the product."""
    n_manifolds = random.randint(2, 5)
    n = random.randint(2, 5)
    space = ProductHPDMatricesAndSiegelDisks(n_manifolds=n_manifolds, n=n)

    assert len(space.factors) == n_manifolds
    assert isinstance(space.factors[0], HPDMatrices)
    assert isinstance(space.factors[0].metric.underlying_metric, HPDAffineMetric)
    assert all(isinstance(factor, Siegel) for factor in space.factors[1:])
    assert all(factor.n == n for factor in space.factors)
    assert [factor.metric.scale for factor in space.factors] == [
        float(n_manifolds - i) for i in range(n_manifolds)
    ]
    assert space.shape == (n_manifolds, n, n)
    assert isinstance(space.metric, ProductHPDMatricesAndSiegelDisksMetric)
