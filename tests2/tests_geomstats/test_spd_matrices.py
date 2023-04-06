import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import (
    SPDAffineMetric,
    SPDBuresWassersteinMetric,
    SPDEuclideanMetric,
    SPDLogEuclideanMetric,
    SPDMatrices,
)
from geomstats.test.comparison import RiemannianMetricComparisonTestCase
from geomstats.test.geometry.spd_matrices import (
    BuresWassersteinBundle,
    BuresWassersteinBundleTestCase,
    SPDAffineMetricTestCase,
    SPDBuresWassersteinMetricTestCase,
    SPDEuclideanMetricTestCase,
    SPDLogEuclideanMetricTestCase,
    SPDMatricesTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from tests2.data.spd_matrices_data import (
    BuresWassersteinBundleTestData,
    SPDAffineMetricTestData,
    SPDBuresWassersteinMetricTestData,
    SPDBuresWassersteinQuotientMetricTestData,
    SPDEuclideanMetricPower1TestData,
    SPDEuclideanMetricTestData,
    SPDLogEuclideanMetricTestData,
    SPDMatricesTestData,
)


class SPDRandomDataGenerator(RandomDataGenerator):
    def __init__(self, space, amplitude=1.0):
        self.space = space
        self.amplitude = amplitude

    def random_tangent_vec(self, base_point):
        return self.space.to_tangent(
            gs.random.normal(size=base_point.shape) / self.amplitude,
            base_point,
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


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def bundle_spaces(request):
    n = request.param
    request.cls.space = BuresWassersteinBundle(n)
    request.cls.base = SPDMatrices(n=n, equip=False)


@pytest.mark.usefixtures("bundle_spaces")
class TestBuresWassersteinBundle(
    BuresWassersteinBundleTestCase, metaclass=DataBasedParametrizer
):
    testing_data = BuresWassersteinBundleTestData()


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

    request.cls.data_generator = SPDRandomDataGenerator(space, amplitude=8.0)


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
def spd_with_quotient_metric(request):
    n = request.param
    space = SPDMatrices(n=n, equip=False)
    request.cls.space = space
    space.equip_with_metric(SPDBuresWassersteinMetric)

    other_space = SPDMatrices(n=n, equip=False)
    request.cls.other_space = other_space
    bundle = BuresWassersteinBundle(n)
    other_space.equip_with_metric(QuotientMetric, fiber_bundle=bundle)

    request.cls.data_generator = SPDRandomDataGenerator(space, amplitude=8.0)


@pytest.mark.usefixtures("spd_with_quotient_metric")
class TestSPDBuresWassersteinQuotientMetric(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SPDBuresWassersteinQuotientMetricTestData()


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

    request.cls.data_generator = SPDRandomDataGenerator(space, amplitude=10.0)


@pytest.mark.usefixtures("spd_with_euclidean_power_1")
class TestSPDEuclideanPower1Metric(
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

    request.cls.data_generator = SPDRandomDataGenerator(space, amplitude=10.0)


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
