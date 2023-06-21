import random

import pytest

from geomstats.geometry.product_positive_reals_and_poincare_disks import (
    ProductPositiveRealsAndComplexPoincareDisks,
    ProductPositiveRealsAndComplexPoincareDisksMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.product_positive_reals_and_poincare_disks import (
    ProductPositiveRealsAndComplexPoincareDisksMetricTestCase,
    ProductPositiveRealsAndComplexPoincareDisksTestCase,
)
from tests2.tests_geomstats.test_geometry.data.product_positive_reals_and_poincare_disks import (
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
    ProductPositiveRealsAndComplexPoincareDisksTestCase, metaclass=DataBasedParametrizer
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
    space = request.cls.space = ProductPositiveRealsAndComplexPoincareDisks(
        n_manifolds=request.param, equip=False
    )
    space.equip_with_metric(ProductPositiveRealsAndComplexPoincareDisksMetric)


@pytest.mark.usefixtures("equipped_spaces")
class TestProductPositiveRealsAndComplexPoincareDisksMetric(
    ProductPositiveRealsAndComplexPoincareDisksMetricTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = ProductPositiveRealsAndComplexPoincareDisksMetricTestData()
