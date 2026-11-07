import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

import geomstats.backend as gs
from geomstats.geometry.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatrices,
)
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
from geomstats.geometry.spd_matrices import (
    CholeskyMap,
    LieCholeskyMetric,
    MatrixPower,
    SPDAffineMetric,
    SPDBuresWassersteinMetric,
    SPDEuclideanMetric,
    SPDLogEuclideanMetric,
    SPDMatrices,
    SPDPowerMetric,
    SymMatrixLog,
)
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.matrices import MatricesMetricTestCase
from geomstats.test_cases.geometry.mixins import ExpDomainTestCaseMixins
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.base import VectorSpaceOpenSetTestData
from .data.diffeo import DiffeoTestData
from .data.riemannian_metric import RiemannianMetricTestData
from .data.spd_matrices import (
    CholeskyMapSmokeTestData,
    MatrixPower05TestData,
    SPD2AffineMetricTestData,
    SPD2BuresWassersteinMetricTestData,
    SPD2EuclideanMetricTestData,
    SPD2LogEuclideanMetricTestData,
    SPD3AffineMetricPower05TestData,
    SPD3BuresWassersteinMetricTestData,
    SPD3EuclideanMetricPower05TestData,
    SPD3EuclideanMetricTestData,
    SPD3LogEuclideanMetricTestData,
    SPDEuclideanMetricTestData,
    SPDMatrices2TestData,
    SPDMatrices3TestData,
    SymMatrixLogSmokeTestData,
)


@pytest.mark.smoke
class TestSymMatrixLogSmoke(DiffeoTestCase, metaclass=DataBasedParametrizer):
    diffeo = SymMatrixLog()
    testing_data = SymMatrixLogSmokeTestData()


class TestSymMatrixLog(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    _n = random.randint(2, 5)
    domain_space = SPDMatrices(n=_n, equip=False)
    image_space = SymmetricMatrices(n=_n, equip=False)
    diffeo = SymMatrixLog()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


@pytest.mark.smoke
class TestMatrixPower05(DiffeoTestCase, metaclass=DataBasedParametrizer):
    diffeo = MatrixPower(power=0.5)
    testing_data = MatrixPower05TestData()


class TestMatrixPower(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    _n = random.randint(2, 5)
    domain_space = image_space = SPDMatrices(n=_n, equip=False)
    diffeo = MatrixPower(power=gs.random.uniform(low=0.5, high=1.5, size=1))

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


@pytest.mark.smoke
class TestCholeskyMapSmoke(DiffeoTestCase, metaclass=DataBasedParametrizer):
    diffeo = CholeskyMap()
    testing_data = CholeskyMapSmokeTestData()


class TestCholeskyMap(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    _n = random.randint(2, 5)
    domain_space = SPDMatrices(n=_n, equip=False)
    image_space = PositiveLowerTriangularMatrices(n=_n, equip=False)
    diffeo = CholeskyMap()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


@pytest.mark.smoke
class TestSPDMatrices2(VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer):
    space = SPDMatrices(n=2, equip=False)
    testing_data = SPDMatrices2TestData()


@pytest.mark.smoke
class TestSPDMatrices3(VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer):
    space = SPDMatrices(n=3, equip=False)
    testing_data = SPDMatrices3TestData()


class TestSPDMatrices(
    VectorSpaceOpenSetTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(2, 5)
    space = SPDMatrices(n=_n, equip=False)
    testing_data = VectorSpaceOpenSetTestData(space)


@pytest.mark.smoke
class TestSPD2AffineMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = SPDMatrices(n=2, equip=False).equip_with_metric(SPDAffineMetric)
    testing_data = SPD2AffineMetricTestData()


@pytest.mark.smoke
class TestSPD3AffineMetricPower05(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    _power = 0.5
    _scale = 1 / _power**2

    image_space = SPDMatrices(n=3, equip=False)
    image_space.equip_with_metric(SPDAffineMetric)
    image_space.equip_with_metric(ScalarProductMetric(image_space, scale=_scale))

    space = SPDMatrices(n=3, equip=False)
    space.equip_with_metric(SPDPowerMetric, image_space=image_space)

    testing_data = SPD3AffineMetricPower05TestData()


class TestSPDAffineMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(2, 5)
    space = SPDMatrices(n=_n, equip=False).equip_with_metric(SPDAffineMetric)
    testing_data = RiemannianMetricTestData(space)


@pytest.mark.smoke
class TestSPD2BuresWassersteinMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SPDMatrices(n=2, equip=False).equip_with_metric(SPDBuresWassersteinMetric)
    testing_data = SPD2BuresWassersteinMetricTestData()


@pytest.mark.smoke
class TestSPD3BuresWassersteinMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SPDMatrices(n=3, equip=False).equip_with_metric(SPDBuresWassersteinMetric)
    testing_data = SPD3BuresWassersteinMetricTestData()


class TestSPDBuresWassersteinMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(2, 5)
    space = SPDMatrices(n=_n, equip=False).equip_with_metric(SPDBuresWassersteinMetric)
    testing_data = RiemannianMetricTestData(space)


@pytest.mark.smoke
class TestSPD2EuclideanMetric(
    ExpDomainTestCaseMixins,
    MatricesMetricTestCase,
    metaclass=DataBasedParametrizer,
):
    space = SPDMatrices(n=2, equip=False).equip_with_metric(SPDEuclideanMetric)
    testing_data = SPD2EuclideanMetricTestData()


@pytest.mark.smoke
class TestSPD3EuclideanMetric(
    ExpDomainTestCaseMixins,
    MatricesMetricTestCase,
    metaclass=DataBasedParametrizer,
):
    space = SPDMatrices(n=3, equip=False).equip_with_metric(SPDEuclideanMetric)
    testing_data = SPD3EuclideanMetricTestData()


@pytest.mark.smoke
class TestSPD3EuclideanMetricPower05(
    ExpDomainTestCaseMixins,
    MatricesMetricTestCase,
    metaclass=DataBasedParametrizer,
):
    _power = 0.5
    _scale = 1 / _power**2

    image_space = SPDMatrices(n=3, equip=False)
    image_space.equip_with_metric(SPDEuclideanMetric)
    image_space.equip_with_metric(ScalarProductMetric(image_space, scale=_scale))

    space = SPDMatrices(n=3, equip=False)
    space.equip_with_metric(SPDPowerMetric, image_space=image_space)

    testing_data = SPD3EuclideanMetricPower05TestData()


@pytest.mark.redundant
class TestSPDEuclideanMetric(
    ExpDomainTestCaseMixins,
    MatricesMetricTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n = random.randint(2, 5)
    space = SPDMatrices(n=_n, equip=False).equip_with_metric(SPDEuclideanMetric)
    testing_data = SPDEuclideanMetricTestData(
        space, generator_kwargs={"amplitude": 4.0}
    )


@pytest.mark.smoke
class TestSPD2LogEuclideanMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SPDMatrices(n=2, equip=False).equip_with_metric(SPDLogEuclideanMetric)
    testing_data = SPD2LogEuclideanMetricTestData()


@pytest.mark.smoke
class TestSPD3LogEuclideanMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = SPDMatrices(n=3, equip=False).equip_with_metric(SPDLogEuclideanMetric)
    testing_data = SPD3LogEuclideanMetricTestData()


@pytest.mark.redundant
class TestSPDLogEuclideanMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(2, 5)
    space = SPDMatrices(n=_n, equip=False).equip_with_metric(SPDLogEuclideanMetric)
    testing_data = RiemannianMetricTestData(space)


@pytest.mark.slow
@pytest.mark.redundant
class TestLieCholeskyMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(2, 3)
    space = SPDMatrices(n=_n, equip=False).equip_with_metric(LieCholeskyMetric)
    testing_data = RiemannianMetricTestData(
        space,
        excluded_methods=("injectivity_radius",),
        generator_kwargs={"amplitude": 10.0},
        sample_locally=True,
    )
