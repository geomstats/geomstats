import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.information_geometry.binomial import (
    BinomialDistributions,
    BinomialDistributionsRandomVariable,
    BinomialMetric,
)
from geomstats.test.test_case import autodiff_only
from geomstats.test_cases.geometry.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestCase,
    CoordinateRiemannianMetricTestCase,
    RiemannianMetricTestCase,
)
from geomstats.test_cases.information_geometry.binomial import (
    BinomialDistributionsTestCase,
)
from geomstats.variants.information_geometry.binomial import (
    BinomialMetric as MetricMatrixBinomialMetric,
)
from geomstats.variants.information_geometry.binomial import PullbackBinomialMetric

from ..test_geometry.data.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestData,
    CoordinateRiemannianMetricTestData,
)
from .data.binomial import (
    Binomial5MetricTestData,
    Binomial7MetricTestData,
    Binomial10MetricTestData,
    BinomialDistributionsTestData,
)


class TestBinomialDistributions(
    BinomialDistributionsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = BinomialDistributions(n_draws=random.randint(2, 5), equip=False)
    random_variable = BinomialDistributionsRandomVariable(space)
    testing_data = BinomialDistributionsTestData(space)


@pytest.mark.smoke
class TestBinomial5Metric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = BinomialDistributions(n_draws=5, equip=False)
    space.equip_with_metric(BinomialMetric)
    testing_data = Binomial5MetricTestData()


@pytest.mark.smoke
class TestBinomial7Metric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = BinomialDistributions(n_draws=7, equip=False)
    space.equip_with_metric(BinomialMetric)
    testing_data = Binomial7MetricTestData()


@pytest.mark.smoke
class TestBinomial10Metric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = BinomialDistributions(n_draws=10, equip=False)
    space.equip_with_metric(BinomialMetric)
    testing_data = Binomial10MetricTestData()


class TestBinomialMetric(
    CoordinateRiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = BinomialDistributions(n_draws=random.randint(2, 5))
    testing_data = CoordinateRiemannianMetricTestData(
        space,
        excluded_methods=("sectional_curvature",),
        generator_kwargs={"within_injectivity_radius": True},
    )


@autodiff_only
@pytest.mark.redundant
class TestBinomialMetricCmp(
    CoordinateRiemannianMetricComparisonTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n_draws = random.randint(2, 5)

    space = BinomialDistributions(n_draws=_n_draws)
    other_space = BinomialDistributions(
        n_draws=_n_draws, equip=False
    ).equip_with_metric(MetricMatrixBinomialMetric)

    testing_data = CoordinateRiemannianMetricComparisonTestData(
        space,
        included_methods=(
            "christoffels",
            "inner_product_derivative_matrix",
            "jacobian_christoffels",
            "curvature",
            "scalar_curvature",
        ),
    )


@autodiff_only
@pytest.mark.redundant
class TestPullbackBinomialMetricCmp(
    CoordinateRiemannianMetricComparisonTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n_draws = random.randint(2, 5)

    space = BinomialDistributions(n_draws=_n_draws)
    other_space = BinomialDistributions(
        n_draws=_n_draws, equip=False
    ).equip_with_metric(PullbackBinomialMetric)

    testing_data = CoordinateRiemannianMetricComparisonTestData(
        space,
        included_methods=(
            "metric_matrix",
            "christoffels",
            "inner_product_derivative_matrix",
            "jacobian_christoffels",
            "curvature",
            "scalar_curvature",
        ),
    )
