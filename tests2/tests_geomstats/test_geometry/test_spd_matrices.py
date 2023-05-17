import random

import pytest

from geomstats.geometry.spd_matrices import (
    SPDAffineMetric,
    SPDBuresWassersteinMetric,
    SPDEuclideanMetric,
    SPDLogEuclideanMetric,
    SPDMatrices,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.spd_matrices import (
    SPDAffineMetricTestCase,
    SPDBuresWassersteinMetricTestCase,
    SPDEuclideanMetricTestCase,
    SPDLogEuclideanMetricTestCase,
    SPDMatricesTestCase,
)

from .data.spd_matrices import (
    SPDAffineMetricTestData,
    SPDBuresWassersteinMetricTestData,
    SPDEuclideanMetricPower1TestData,
    SPDEuclideanMetricTestData,
    SPDLogEuclideanMetricTestData,
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
        (2, 1),
        (random.randint(3, 5), 1),
        (2, 0.5),
        (random.randint(3, 5), 0.5),
        (2, -0.5),
        (random.randint(3, 5), -0.5),
    ],
)
def spd_with_affine_metric(request):
    n, power_affine = request.param
    space = SPDMatrices(n=n, equip=False)
    request.cls.space = space
    space.equip_with_metric(SPDAffineMetric, power_affine=power_affine)


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

    request.cls.data_generator = RandomDataGenerator(space, amplitude=2.0)


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
def spd_with_euclidean_power_1(request):
    n = request.param

    space = SPDMatrices(n=n, equip=False)
    request.cls.space = space
    space.equip_with_metric(SPDEuclideanMetric)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=4.0)


@pytest.mark.usefixtures("spd_with_euclidean_power_1")
class TestSPDEuclideanMetricPower1(
    SPDEuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDEuclideanMetricPower1TestData()


@pytest.fixture(
    scope="class",
    params=[
        (2, -0.5),
        (random.randint(3, 5), -0.5),
        (2, 0.5),
        (random.randint(3, 5), 0.5),
    ],
)
def spd_with_euclidean(request):
    n, power_euclidean = request.param

    space = SPDMatrices(n=n, equip=False)
    request.cls.space = space
    space.equip_with_metric(SPDEuclideanMetric, power_euclidean=power_euclidean)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=2.0)


@pytest.mark.usefixtures("spd_with_euclidean")
class TestSPDEuclideanMetric(
    SPDEuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDEuclideanMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spd_with_log_euclidean(request):
    n = request.param
    space = SPDMatrices(n=n, equip=False)
    request.cls.space = space
    space.equip_with_metric(SPDLogEuclideanMetric)


@pytest.mark.usefixtures("spd_with_log_euclidean")
class TestSPDLogEuclideanMetric(
    SPDLogEuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDLogEuclideanMetricTestData()
