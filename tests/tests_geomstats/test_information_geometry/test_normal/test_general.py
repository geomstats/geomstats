import random

import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.information_geometry.normal import NormalDistributions
from geomstats.information_geometry.normal.general import (
    MultivariateNormalDistributionsRandomVariable,
)
from geomstats.test_cases.geometry.product_manifold import ProductManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)

from ...test_geometry.data.riemannian_metric import RiemannianMetricTestData
from ..data.normal.general import GeneralNormalDistributionsTestData


class TestGeneralNormalDistributions(
    InformationManifoldMixinTestCase,
    ProductManifoldTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = NormalDistributions(sample_dim=random.randint(3, 5), equip=False)
    random_variable = MultivariateNormalDistributionsRandomVariable(space)
    testing_data = GeneralNormalDistributionsTestData(
        space, excluded_methods=("sample_belongs_to_support",)
    )


@pytest.mark.redundant
class TestGeneralNormalMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = NormalDistributions(sample_dim=random.randint(3, 5))
    testing_data = RiemannianMetricTestData(space)
