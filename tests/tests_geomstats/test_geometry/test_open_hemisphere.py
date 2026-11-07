import random

import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.open_hemisphere import (
    OpenHemisphere,
    OpenHemispheresProduct,
    OpenHemispheresProductMetric,
    OpenHemisphereToHyperboloidDiffeo,
)
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.product_manifold import ProductManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.base import OpenSetTestData
from .data.diffeo import DiffeoTestData
from .data.product_manifold import ProductManifoldTestData
from .data.riemannian_metric import RiemannianMetricTestData


class TestOpenHemisphereToHyperboloidDiffeo(
    DiffeoTestCase, metaclass=GeometricDataBasedParametrizer
):
    _dim = random.randint(1, 4)
    domain_space = OpenHemisphere(_dim, equip=False)
    image_space = Hyperboloid(_dim, equip=False)
    diffeo = OpenHemisphereToHyperboloidDiffeo()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestOpenHemisphere(OpenSetTestCase, metaclass=GeometricDataBasedParametrizer):
    space = OpenHemisphere(random.randint(1, 4), equip=False)
    testing_data = OpenSetTestData(space)


class TestOpenHemispherePullbackMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = OpenHemisphere(random.randint(1, 4))
    testing_data = RiemannianMetricTestData(space)


@pytest.mark.redundant
class TestOpenHemispheresProduct(
    ProductManifoldTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = OpenHemispheresProduct(random.randint(3, 5), equip=False)
    testing_data = ProductManifoldTestData(space)


@pytest.mark.redundant
class TestOpenHemispheresProductMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = OpenHemispheresProduct(random.randint(3, 5))
    testing_data = RiemannianMetricTestData(space, sample_locally=True)


@pytest.mark.smoke
def test_open_hemispheres_product_structure():
    """Check the factors, dimension, shape and metric of the product."""
    n = random.randint(2, 5)
    space = OpenHemispheresProduct(n)

    assert [factor.dim for factor in space.factors] == list(range(1, n))
    assert space.dim == n * (n - 1) // 2
    assert space.shape == (n * (n + 1) // 2 - 1,)
    assert isinstance(space.metric, OpenHemispheresProductMetric)
