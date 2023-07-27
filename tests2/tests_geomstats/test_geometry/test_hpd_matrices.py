import random

import pytest

from geomstats.geometry.hpd_matrices import (
    HPDAffineMetric,
    HPDBuresWassersteinMetric,
    HPDEuclideanMetric,
    HPDLogEuclideanMetric,
    HPDMatrices,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import ComplexOpenSetTestCase
from geomstats.test_cases.geometry.complex_riemannian_metric import (
    ComplexRiemannianMetricTestCase,
)
from geomstats.test_cases.geometry.spd_matrices import SPDMatricesTestCaseMixins

from .data.hpd_matrices import (
    HPDAffineMetricPower1TestData,
    HPDAffineMetricTestData,
    HPDBuresWassersteinMetricTestData,
    HPDEuclideanMetricTestData,
    HPDLogEuclideanMetricTestData,
    HPDMatrices2TestData,
    HPDMatrices3TestData,
    HPDMatricesTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = HPDMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestHPDMatrices(
    SPDMatricesTestCaseMixins, ComplexOpenSetTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HPDMatricesTestData()


@pytest.mark.smoke
class TestHPDMatrices2(ComplexOpenSetTestCase, metaclass=DataBasedParametrizer):
    space = HPDMatrices(n=2, equip=False)
    testing_data = HPDMatrices2TestData()


@pytest.mark.smoke
class TestHPDMatrices3(ComplexOpenSetTestCase, metaclass=DataBasedParametrizer):
    space = HPDMatrices(n=3, equip=False)
    testing_data = HPDMatrices3TestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces_with_affine_metric_power_1(request):
    n = request.param
    space = request.cls.space = HPDMatrices(n=n, equip=False)
    space.equip_with_metric(HPDAffineMetric, power_affine=1)


@pytest.mark.usefixtures("spaces_with_affine_metric_power_1")
class TestHPDAffineMetricPower1(
    ComplexRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HPDAffineMetricPower1TestData()


@pytest.fixture(
    scope="class",
    params=[
        (2, 0.5),
        (random.randint(3, 5), 0.5),
        (2, -0.5),
        (random.randint(3, 5), -0.5),
    ],
)
def spaces_with_affine_metric(request):
    n, power_affine = request.param
    space = HPDMatrices(n=n, equip=False)
    request.cls.space = space
    space.equip_with_metric(HPDAffineMetric, power_affine=power_affine)


@pytest.mark.usefixtures("spaces_with_affine_metric")
class TestHPDAffineMetric(
    ComplexRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HPDAffineMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces_with_bw_metric(request):
    space = HPDMatrices(n=request.param, equip=False)
    request.cls.space = space
    space.equip_with_metric(HPDBuresWassersteinMetric)


@pytest.mark.redundant
@pytest.mark.usefixtures("spaces_with_bw_metric")
class TestHPDBuresWassersteinMetric(
    ComplexRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HPDBuresWassersteinMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        (2, 1),
        (random.randint(3, 5), 1),
        (2, -0.5),
        (random.randint(3, 5), -0.5),
        (2, 0.5),
        (random.randint(3, 5), 0.5),
    ],
)
def hpd_with_euclidean_metric(request):
    n, power_euclidean = request.param

    space = HPDMatrices(n=n, equip=False)
    request.cls.space = space
    space.equip_with_metric(HPDEuclideanMetric, power_euclidean=power_euclidean)


@pytest.mark.usefixtures("hpd_with_euclidean_metric")
class TestHPDEuclideanMetric(
    ComplexRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HPDEuclideanMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces_with_log_euclidean(request):
    n = request.param
    space = HPDMatrices(n=n, equip=False)
    request.cls.space = space
    space.equip_with_metric(HPDLogEuclideanMetric)


@pytest.mark.usefixtures("spaces_with_log_euclidean")
class TestHPDLogEuclideanMetric(
    ComplexRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HPDLogEuclideanMetricTestData()
