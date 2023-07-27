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
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.geometry.spd_matrices import (
    SPDEuclideanMetricTestCase,
    SPDMatricesTestCase,
)

from .data.spd_matrices import (
    SPD2AffineMetricPower1TestData,
    SPD2BuresWassersteinMetricTestData,
    SPD2EuclideanMetricPower1TestData,
    SPD2LogEuclideanMetricTestData,
    SPD3AffineMetricPower05TestData,
    SPD3BuresWassersteinMetricTestData,
    SPD3EuclideanMetricPower1TestData,
    SPD3EuclideanMetricPower05TestData,
    SPD3LogEuclideanMetricTestData,
    SPDAffineMetricPower1TestData,
    SPDAffineMetricTestData,
    SPDBuresWassersteinMetricTestData,
    SPDEuclideanMetricPower1TestData,
    SPDEuclideanMetricTestData,
    SPDLogEuclideanMetricTestData,
    SPDMatrices2TestData,
    SPDMatrices3TestData,
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


@pytest.mark.smoke
class TestSPDMatrices2(SPDMatricesTestCase, metaclass=DataBasedParametrizer):
    space = SPDMatrices(n=2, equip=False)
    testing_data = SPDMatrices2TestData()


@pytest.mark.smoke
class TestSPDMatrices3(SPDMatricesTestCase, metaclass=DataBasedParametrizer):
    space = SPDMatrices(n=3, equip=False)
    testing_data = SPDMatrices3TestData()


@pytest.fixture(
    scope="class",
    params=[
        (2, 1),
        (random.randint(3, 5), 1),
    ],
)
def spd_with_affine_metric_power_1(request):
    n, power_affine = request.param
    space = SPDMatrices(n=n, equip=False)
    request.cls.space = space
    space.equip_with_metric(SPDAffineMetric, power_affine=power_affine)


@pytest.mark.usefixtures("spd_with_affine_metric_power_1")
class TestSPDAffineMetricPower1(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDAffineMetricPower1TestData()


@pytest.fixture(
    scope="class",
    params=[
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
class TestSPDAffineMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = SPDAffineMetricTestData()


@pytest.mark.smoke
class TestSPD2AffineMetricPower1(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPD2AffineMetricPower1TestData()
    space = SPDMatrices(n=2, equip=False)
    space.equip_with_metric(SPDAffineMetric, power_affine=1)


@pytest.mark.smoke
class TestSPD3AffineMetricPower05(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPD3AffineMetricPower05TestData()
    space = SPDMatrices(n=3, equip=False)
    space.equip_with_metric(SPDAffineMetric, power_affine=0.5)


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


@pytest.mark.redundant
@pytest.mark.usefixtures("spd_with_bw_metric")
class TestSPDBuresWassersteinMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDBuresWassersteinMetricTestData()


@pytest.mark.smoke
class TestSPD2BuresWassersteinMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPD2BuresWassersteinMetricTestData()
    space = SPDMatrices(n=2, equip=False)
    space.equip_with_metric(SPDBuresWassersteinMetric)


@pytest.mark.smoke
class TestSPD3BuresWassersteinMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPD3BuresWassersteinMetricTestData()
    space = SPDMatrices(n=3, equip=False)
    space.equip_with_metric(SPDBuresWassersteinMetric)


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


@pytest.mark.smoke
class TestSPD2EuclideanMetricPower1(
    SPDEuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SPDMatrices(n=2, equip=False)
    space.equip_with_metric(SPDEuclideanMetric, power_euclidean=1)
    testing_data = SPD2EuclideanMetricPower1TestData()


@pytest.mark.smoke
class TestSPD3EuclideanMetricPower1(
    SPDEuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SPDMatrices(n=3, equip=False)
    space.equip_with_metric(SPDEuclideanMetric, power_euclidean=1)
    testing_data = SPD3EuclideanMetricPower1TestData()


@pytest.mark.smoke
class TestSPD3EuclideanMetricPower05(
    SPDEuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SPDMatrices(n=2, equip=False)
    space.equip_with_metric(SPDEuclideanMetric, power_euclidean=0.5)
    testing_data = SPD3EuclideanMetricPower05TestData()


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
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDLogEuclideanMetricTestData()


@pytest.mark.smoke
class TestSPD2LogEuclideanMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SPDMatrices(n=2, equip=False)
    space.equip_with_metric(SPDLogEuclideanMetric)
    testing_data = SPD2LogEuclideanMetricTestData()


@pytest.mark.smoke
class TestSPD3LogEuclideanMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SPDMatrices(n=3, equip=False)
    space.equip_with_metric(SPDLogEuclideanMetric)
    testing_data = SPD3LogEuclideanMetricTestData()
