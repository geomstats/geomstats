import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

import geomstats.backend as gs
from geomstats.geometry.hermitian_matrices import HermitianMatrices
from geomstats.geometry.hpd_matrices import (
    HPDAffineMetric,
    HPDBuresWassersteinMetric,
    HPDEuclideanMetric,
    HPDLogEuclideanMetric,
    HPDMatrices,
)
from geomstats.geometry.spd_matrices import MatrixPower, SymMatrixLog
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.mixins import ExpDomainTestCaseMixins
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.base import VectorSpaceOpenSetTestData
from .data.diffeo import DiffeoTestData
from .data.hpd_matrices import (
    HPDEuclideanMetricTestData,
    HPDMatrices2TestData,
    HPDMatrices3TestData,
)
from .data.riemannian_metric import RiemannianMetricTestData


class TestSymMatrixLog(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    _n = random.randint(2, 5)
    domain_space = HPDMatrices(n=_n, equip=False)
    image_space = HermitianMatrices(n=_n, equip=False)
    diffeo = SymMatrixLog()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestMatrixPower(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    _n = random.randint(2, 5)
    domain_space = image_space = HPDMatrices(n=_n, equip=False)
    diffeo = MatrixPower(power=gs.random.uniform(size=1))

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestHPDMatrices(
    VectorSpaceOpenSetTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = HPDMatrices(n=random.randint(2, 5), equip=False)
    testing_data = VectorSpaceOpenSetTestData(space)


@pytest.mark.smoke
class TestHPDMatrices2(VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer):
    space = HPDMatrices(n=2, equip=False)
    testing_data = HPDMatrices2TestData()


@pytest.mark.smoke
class TestHPDMatrices3(VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer):
    space = HPDMatrices(n=3, equip=False)
    testing_data = HPDMatrices3TestData()


class TestHPDAffineMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = HPDMatrices(n=random.randint(2, 5), equip=False).equip_with_metric(
        HPDAffineMetric
    )
    testing_data = RiemannianMetricTestData(space)


class TestHPDBuresWassersteinMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = HPDMatrices(n=random.randint(2, 5), equip=False).equip_with_metric(
        HPDBuresWassersteinMetric
    )
    testing_data = RiemannianMetricTestData(
        space,
        generator_kwargs={"amplitude": 4.0},
    )


class TestHPDEuclideanMetric(
    ExpDomainTestCaseMixins,
    RiemannianMetricTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = HPDMatrices(n=random.randint(2, 5), equip=False).equip_with_metric(
        HPDEuclideanMetric
    )
    testing_data = HPDEuclideanMetricTestData(
        space, generator_kwargs={"amplitude": 5.0}
    )


class TestHPDLogEuclideanMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = HPDMatrices(n=random.randint(2, 5), equip=False).equip_with_metric(
        HPDLogEuclideanMetric
    )
    testing_data = RiemannianMetricTestData(
        space, excluded_methods=("injectivity_radius",)
    )
