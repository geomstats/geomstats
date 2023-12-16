import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.hermitian_matrices import HermitianMatrices
from geomstats.geometry.hpd_matrices import (
    HPDBuresWassersteinMetric,
    HPDEuclideanMetric,
    HPDLogEuclideanMetric,
    HPDMatrices,
)
from geomstats.geometry.spd_matrices import MatrixPower, SymMatrixLog
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.base import ComplexVectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.complex_riemannian_metric import (
    ComplexRiemannianMetricTestCase,
)
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase

from .data.base import ComplexVectorSpaceOpenSetTestData
from .data.diffeo import DiffeoTestData
from .data.hpd_matrices import (
    HPDAffineMetricTestData,
    HPDBuresWassersteinMetricTestData,
    HPDEuclideanMetricTestData,
    HPDLogEuclideanMetricTestData,
    HPDMatrices2TestData,
    HPDMatrices3TestData,
)


class TestSymMatrixLog(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)
    space = HPDMatrices(n=_n, equip=False)
    image_space = HermitianMatrices(n=_n, equip=False)
    diffeo = SymMatrixLog()
    testing_data = DiffeoTestData()


class TestMatrixPower(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)
    space = image_space = HPDMatrices(n=_n, equip=False)
    diffeo = MatrixPower(power=gs.random.uniform(size=1))
    testing_data = DiffeoTestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
    ],
)
def spaces(request):
    request.cls.space = HPDMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestHPDMatrices(
    ComplexVectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ComplexVectorSpaceOpenSetTestData()


@pytest.mark.smoke
class TestHPDMatrices2(
    ComplexVectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer
):
    space = HPDMatrices(n=2, equip=False)
    testing_data = HPDMatrices2TestData()


@pytest.mark.smoke
class TestHPDMatrices3(
    ComplexVectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer
):
    space = HPDMatrices(n=3, equip=False)
    testing_data = HPDMatrices3TestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
    ],
)
def spaces_with_affine_metric(request):
    n = request.param
    request.cls.space = HPDMatrices(n=n)


@pytest.mark.usefixtures("spaces_with_affine_metric")
class TestHPDAffineMetric(
    ComplexRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HPDAffineMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
    ],
)
def spaces_with_bw_metric(request):
    space = request.cls.space = HPDMatrices(n=request.param, equip=False)
    space.equip_with_metric(HPDBuresWassersteinMetric)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=4.0)


@pytest.mark.usefixtures("spaces_with_bw_metric")
class TestHPDBuresWassersteinMetric(
    ComplexRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HPDBuresWassersteinMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
    ],
)
def hpd_with_euclidean_metric(request):
    n = request.param

    space = request.cls.space = HPDMatrices(n=n, equip=False)
    space.equip_with_metric(HPDEuclideanMetric)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=5.0)


@pytest.mark.usefixtures("hpd_with_euclidean_metric")
class TestHPDEuclideanMetric(
    ComplexRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HPDEuclideanMetricTestData()

    def test_exp_domain(self, tangent_vec, base_point, expected, atol):
        res = self.space.metric.exp_domain(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
    ],
)
def spaces_with_log_euclidean(request):
    n = request.param
    space = request.cls.space = HPDMatrices(n=n, equip=False)
    space.equip_with_metric(HPDLogEuclideanMetric)


@pytest.mark.usefixtures("spaces_with_log_euclidean")
class TestHPDLogEuclideanMetric(
    PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HPDLogEuclideanMetricTestData()
