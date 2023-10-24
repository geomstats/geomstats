import random

import pytest

from geomstats.geometry.product_positive_reals_and_poincare_disks import (
    ProductPositiveRealsAndComplexPoincareDisks,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.product_manifold import ProductManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.product_positive_reals_and_poincare_disks import (
    ProductPositiveRealsAndComplexPoincareDisksMetricTestData,
    ProductPositiveRealsAndComplexPoincareDisksTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = ProductPositiveRealsAndComplexPoincareDisks(
        n_manifolds=request.param, equip=False
    )


@pytest.mark.usefixtures("spaces")
class TestProductPositiveRealsAndComplexPoincareDisks(
    ProductManifoldTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ProductPositiveRealsAndComplexPoincareDisksTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_spaces(request):
    request.cls.space = ProductPositiveRealsAndComplexPoincareDisks(
        n_manifolds=request.param
    )


@pytest.mark.usefixtures("equipped_spaces")
class TestProductPositiveRealsAndComplexPoincareDisksMetric(
    RiemannianMetricTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = ProductPositiveRealsAndComplexPoincareDisksMetricTestData()
