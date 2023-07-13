import random

import pytest

from geomstats.geometry.product_hpd_and_siegel_disks import (
    ProductHPDMatricesAndSiegelDisks,
    ProductHPDMatricesAndSiegelDisksMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.product_manifold import ProductManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from tests2.tests_geomstats.test_geometry.data.product_hpd_and_siegel_disks import (
    ProductHPDMatricesAndSiegelDisksMetricTestData,
    ProductHPDMatricesAndSiegelDisksTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        (2, 3),
        (random.randint(3, 5), random.randint(3, 5)),
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
        (2, 3),
        (random.randint(3, 5), random.randint(3, 5)),
    ],
)
def equipped_spaces(request):
    n_manifolds, n = request.param
    space = request.cls.space = ProductHPDMatricesAndSiegelDisks(
        n_manifolds=n_manifolds, n=n, equip=False
    )

    space.equip_with_metric(ProductHPDMatricesAndSiegelDisksMetric)


@pytest.mark.usefixtures("equipped_spaces")
class TestProductHPDMatricesAndSiegelDisksMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ProductHPDMatricesAndSiegelDisksMetricTestData()
