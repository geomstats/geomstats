import random

import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.information_geometry.normal import NormalDistributions
from geomstats.information_geometry.normal.centered import (
    SharedMeanNormalDistributionsRandomVariable,
)
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)

from ...test_geometry.data.spd_matrices import SPDAffineMetricTestData
from ..data.normal.centered import CenteredNormalDistributionsTestData


class TestCenteredNormalDistributions(
    InformationManifoldMixinTestCase,
    VectorSpaceOpenSetTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = NormalDistributions(
        sample_dim=random.randint(3, 5), distribution_type="centered", equip=False
    )
    random_variable = SharedMeanNormalDistributionsRandomVariable(space)
    testing_data = CenteredNormalDistributionsTestData(
        space, excluded_methods=("sample_belongs_to_support",)
    )


@pytest.mark.redundant
class TestCenteredNormalMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = NormalDistributions(
        sample_dim=random.randint(3, 5), distribution_type="centered"
    )
    testing_data = SPDAffineMetricTestData(space)
