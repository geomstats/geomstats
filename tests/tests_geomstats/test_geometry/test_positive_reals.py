import random

import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.positive_reals import (
    LogDiffeo,
    PositiveReals,
    PowerDiffeo,
    PowerMetric,
)
from geomstats.test.test_case import autodiff_only
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestCase,
    CoordinateRiemannianMetricTestCase,
)
from geomstats.variants.geometry.positive_reals import (
    LogMetric as MetricMatrixLogMetric,
)
from geomstats.variants.geometry.positive_reals import (
    PowerMetric as MetricMatrixPowerMetric,
)
from geomstats.variants.geometry.positive_reals import (
    PullbackLogMetric,
    PullbackPowerMetric,
)

from .data.diffeo import DiffeoTestData
from .data.positive_reals import (
    LogDiffeoTestData,
    LogMetricTestData,
    PositiveRealsTestData,
)
from .data.pullback_metric import get_coordinate_pullback_excluded_methods
from .data.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestData,
    CoordinateRiemannianMetricTestData,
)


class TestLogDiffeo(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    domain_space = PositiveReals(equip=False)
    image_space = Euclidean(dim=1, equip=False)
    diffeo = LogDiffeo()

    testing_data = LogDiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestPowerDiffeo(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    domain_space = PositiveReals(equip=False)
    image_space = PositiveReals(equip=False)
    diffeo = PowerDiffeo(random.uniform(0.5, 2.0))

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestPositiveReals(
    VectorSpaceOpenSetTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PositiveReals(equip=False)
    testing_data = PositiveRealsTestData(space)


class TestLogMetric(
    CoordinateRiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PositiveReals()
    testing_data = LogMetricTestData(
        space,
        excluded_methods=("sectional_curvature",),
        generator_kwargs={"bounded_metric_norm": True},
    )


@autodiff_only
@pytest.mark.redundant
class TestLogMetricCmp(
    CoordinateRiemannianMetricComparisonTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = PositiveReals()
    other_space = PositiveReals(equip=False).equip_with_metric(MetricMatrixLogMetric)
    testing_data = CoordinateRiemannianMetricComparisonTestData(
        space,
        included_methods=(
            "christoffels",
            "inner_product_derivative_matrix",
            "jacobian_christoffels",
            "curvature",
            "scalar_curvature",
        ),
        generator_kwargs={"bounded_metric_norm": True},
    )


class TestPowerMetric(
    CoordinateRiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PositiveReals(equip=False).equip_with_metric(
        PowerMetric, power=random.uniform(0.5, 2.0)
    )
    testing_data = CoordinateRiemannianMetricTestData(
        space,
        excluded_methods=("sectional_curvature",),
        generator_kwargs={
            "bounded_metric_norm": True,
            "within_injectivity_radius": True,
        },
    )


@autodiff_only
@pytest.mark.redundant
class TestPowerMetricCmp(
    CoordinateRiemannianMetricComparisonTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _power = random.uniform(0.5, 2.0)
    space = PositiveReals(equip=False).equip_with_metric(PowerMetric, power=_power)
    other_space = PositiveReals(equip=False).equip_with_metric(
        MetricMatrixPowerMetric, power=_power
    )
    testing_data = CoordinateRiemannianMetricComparisonTestData(
        space,
        included_methods=(
            "christoffels",
            "inner_product_derivative_matrix",
            "jacobian_christoffels",
            "curvature",
            "scalar_curvature",
        ),
        generator_kwargs={
            "bounded_metric_norm": True,
            "within_injectivity_radius": True,
        },
    )


@pytest.mark.redundant
class TestPullbackLogMetric(
    CoordinateRiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PositiveReals(equip=False).equip_with_metric(PullbackLogMetric)
    testing_data = LogMetricTestData(
        space,
        excluded_methods=get_coordinate_pullback_excluded_methods()
        + ("sectional_curvature",),
        generator_kwargs={"bounded_metric_norm": True},
    )


@autodiff_only
@pytest.mark.redundant
class TestPullbackLogMetricCmp(
    CoordinateRiemannianMetricComparisonTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = PositiveReals()
    other_space = PositiveReals(equip=False).equip_with_metric(PullbackLogMetric)
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
        generator_kwargs={"bounded_metric_norm": True},
    )


@autodiff_only
@pytest.mark.redundant
class TestPullbackPowerMetricCmp(
    CoordinateRiemannianMetricComparisonTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _power = random.uniform(0.5, 2.0)
    space = PositiveReals(equip=False).equip_with_metric(PowerMetric, power=_power)
    other_space = PositiveReals(equip=False).equip_with_metric(
        PullbackPowerMetric, power=_power
    )
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
        generator_kwargs={
            "bounded_metric_norm": True,
            "within_injectivity_radius": True,
        },
    )
