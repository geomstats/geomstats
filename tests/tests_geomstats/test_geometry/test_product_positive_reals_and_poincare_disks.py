import random

import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.complex_poincare_disk import ComplexPoincareDisk
from geomstats.geometry.positive_reals import PositiveReals
from geomstats.geometry.product_positive_reals_and_poincare_disks import (
    ProductPositiveRealsAndComplexPoincareDisks,
    ProductPositiveRealsAndComplexPoincareDisksMetric,
)
from geomstats.test_cases.geometry.product_manifold import ProductManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.product_manifold import ProductManifoldTestData
from .data.riemannian_metric import RiemannianMetricTestData


@pytest.mark.redundant
class TestProductPositiveRealsAndComplexPoincareDisks(
    ProductManifoldTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ProductPositiveRealsAndComplexPoincareDisks(
        n_manifolds=random.randint(2, 5), equip=False
    )
    testing_data = ProductManifoldTestData(space)


@pytest.mark.redundant
class TestProductPositiveRealsAndComplexPoincareDisksMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ProductPositiveRealsAndComplexPoincareDisks(
        n_manifolds=random.randint(2, 5)
    )
    testing_data = RiemannianMetricTestData(
        space,
        method_defaults={"geodesic_ivp_belongs": {"atol": 1e-6}},
    )


@pytest.mark.smoke
def test_product_positive_reals_and_complex_poincare_disks_structure():
    """Check the factors, scales, dimension, shape and metric of the product."""
    n = random.randint(2, 5)
    space = ProductPositiveRealsAndComplexPoincareDisks(n_manifolds=n)

    assert len(space.factors) == n
    assert isinstance(space.factors[0], PositiveReals)
    assert all(isinstance(factor, ComplexPoincareDisk) for factor in space.factors[1:])
    assert [factor.metric.scale for factor in space.factors] == [
        float(n - i) for i in range(n)
    ]
    assert space.dim == n
    assert space.shape == (n, 1)
    assert isinstance(space.metric, ProductPositiveRealsAndComplexPoincareDisksMetric)
