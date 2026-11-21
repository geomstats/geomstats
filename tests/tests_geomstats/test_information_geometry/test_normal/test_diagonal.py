import random

from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.information_geometry.normal import NormalDistributions
from geomstats.information_geometry.normal.diagonal import (
    DiagonalNormalDistributionsRandomVariable,
)
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)

from ...test_geometry.data.riemannian_metric import RiemannianMetricTestData
from ..data.normal.diagonal import DiagonalNormalDistributionsTestData


class TestDiagonalNormalDistributions(
    InformationManifoldMixinTestCase,
    VectorSpaceOpenSetTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = NormalDistributions(
        sample_dim=random.randint(2, 5), distribution_type="diagonal", equip=False
    )
    random_variable = DiagonalNormalDistributionsRandomVariable(space)
    testing_data = DiagonalNormalDistributionsTestData(
        space, excluded_methods=("sample_belongs_to_support",)
    )


class TestDiagonalNormalMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = NormalDistributions(
        sample_dim=random.randint(2, 5), distribution_type="diagonal"
    )
    testing_data = RiemannianMetricTestData(
        space,
        excluded_methods=(
            "parallel_transport_ivp",
            "parallel_transport_bvp",
            "parallel_transport_ivp_transported_is_tangent",
            "parallel_transport_bvp_transported_is_tangent",
            "parallel_transport_ivp_norm",
            "parallel_transport_bvp_norm",
        ),
        generator_kwargs={"amplitude": 10.0, "bounded_metric_norm": True},
    )
