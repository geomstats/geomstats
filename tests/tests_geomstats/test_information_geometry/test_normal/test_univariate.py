import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.information_geometry.normal import NormalDistributions
from geomstats.information_geometry.normal.univariate import (
    UnivariateNormalDistributionsRandomVariable,
)
from geomstats.test.test_case import autodiff_only
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestCase,
)
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)
from geomstats.test_cases.information_geometry.normal.univariate import (
    UnivariateNormalMetricTestCase,
)
from geomstats.variants.information_geometry.normal.univariate import (
    PullbackUnivariateNormalMetric,
)

from ...test_geometry.data.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestData,
)
from ..data.normal.univariate import (
    UnivariateNormalDistributionsTestData,
    UnivariateNormalMetricTestData,
)


class TestUnivariateNormalDistributions(
    InformationManifoldMixinTestCase,
    VectorSpaceOpenSetTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = NormalDistributions(sample_dim=1, equip=False)
    random_variable = UnivariateNormalDistributionsRandomVariable(space)
    testing_data = UnivariateNormalDistributionsTestData(
        space, excluded_methods=("sample_belongs_to_support",)
    )


class TestUnivariateNormalMetric(
    UnivariateNormalMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = NormalDistributions(sample_dim=1)
    testing_data = UnivariateNormalMetricTestData(
        space,
        generator_kwargs={"amplitude": 5.0, "bounded_metric_norm": True},
    )


@autodiff_only
@pytest.mark.redundant
class TestPullbackUnivariateNormalMetricCmp(
    CoordinateRiemannianMetricComparisonTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = NormalDistributions(sample_dim=1)
    other_space = NormalDistributions(sample_dim=1, equip=False).equip_with_metric(
        PullbackUnivariateNormalMetric
    )
    testing_data = CoordinateRiemannianMetricComparisonTestData(
        space,
        included_methods=(
            "metric_matrix",
            "christoffels",
            "inner_product_derivative_matrix",
            "jacobian_christoffels",
            "riemann_tensor",
            "sectional_curvature",
            "scalar_curvature",
        ),
        generator_kwargs={"bounded_metric_norm": True},
    )
