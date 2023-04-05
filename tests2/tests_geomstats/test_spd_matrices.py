import random

import pytest

from geomstats.geometry.spd_matrices import (
    SPDAffineMetric,
    SPDBuresWassersteinMetric,
    SPDEuclideanMetric,
    SPDMatrices,
)
from geomstats.test.geometry.spd_matrices import (
    SPDAffineMetricTestCase,
    SPDBuresWassersteinMetricTestCase,
    SPDEuclideanMetricTestCase,
    SPDMatricesTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.spd_matrices_data import (
    SPDAffineMetricTestData,
    SPDBuresWassersteinMetricTestData,
    SPDEuclideanMetricTestData,
    SPDMatricesTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = SPDMatrices(n=request.param)


@pytest.mark.usefixtures("spaces")
class TestSPDMatrices(SPDMatricesTestCase, metaclass=DataBasedParametrizer):
    testing_data = SPDMatricesTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spd_with_affine_metric(request):
    space = SPDMatrices(n=request.param, equip=False)
    request.cls.space = space
    space.equip_with_metric(SPDAffineMetric)


@pytest.mark.usefixtures("spd_with_affine_metric")
class TestSPDAffineMetric(SPDAffineMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = SPDAffineMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spd_with_bw_metric(request):
    space = SPDMatrices(n=request.param, equip=False)
    request.cls.space = space
    space.equip_with_metric(SPDBuresWassersteinMetric)


@pytest.mark.usefixtures("spd_with_bw_metric")
class TestSPDBuresWassersteinMetric(
    SPDBuresWassersteinMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDBuresWassersteinMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spd_with_euclidean(request):
    space = SPDMatrices(n=request.param, equip=False)
    request.cls.space = space
    space.equip_with_metric(SPDEuclideanMetric)


@pytest.mark.usefixtures("spd_with_euclidean")
class TestSPDEuclideanMetric(
    SPDEuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDEuclideanMetricTestData()
