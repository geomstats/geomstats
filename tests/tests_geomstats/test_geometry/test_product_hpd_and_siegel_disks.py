import random

import pytest

from geomstats.geometry.product_hpd_and_siegel_disks import (
    ProductHPDMatricesAndSiegelDisks,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.product_manifold import ProductManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.product_hpd_and_siegel_disks import (
    ProductHPDMatricesAndSiegelDisksMetricTestData,
    ProductHPDMatricesAndSiegelDisksTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        (random.randint(2, 5), random.randint(2, 5)),
    ],
)
def spaces(request):
    n_manifolds, n = request.param
    request.cls.space = ProductHPDMatricesAndSiegelDisks(
        n_manifolds=n_manifolds, n=n, equip=False
    )


@pytest.mark.usefixtures("spaces")
class TestProductHPDMatricesAndSiegelDisks(
    ProductManifoldTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ProductHPDMatricesAndSiegelDisksTestData()


@pytest.fixture(
    scope="class",
    params=[
        (random.randint(2, 5), random.randint(2, 5)),
    ],
)
def equipped_spaces(request):
    n_manifolds, n = request.param
    request.cls.space = ProductHPDMatricesAndSiegelDisks(n_manifolds=n_manifolds, n=n)


@pytest.mark.usefixtures("equipped_spaces")
class TestProductHPDMatricesAndSiegelDisksMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ProductHPDMatricesAndSiegelDisksMetricTestData()
