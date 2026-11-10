import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.poincare_ball import PoincareBall, PoincareBallMetric
from geomstats.test_cases.geometry.poincare_ball import (
    PoincareBallMetricTestCase,
    PoincareBallTestCase,
)
from geomstats.test_cases.geometry.retraction import RetractionTestCase

from .data.poincare_ball import (
    PoincareBall2MetricTestData,
    PoincareBall2TestData,
    PoincareBallMetricTestData,
    PoincareBallTestData,
)
from .data.retraction import RetractionTestData


@pytest.mark.smoke
class TestPoincareBall2(PoincareBallTestCase, metaclass=DataBasedParametrizer):
    space = PoincareBall(dim=2, equip=False)
    testing_data = PoincareBall2TestData()


class TestPoincareBall(PoincareBallTestCase, metaclass=GeometricDataBasedParametrizer):
    space = PoincareBall(dim=random.randint(2, 5), equip=False)
    testing_data = PoincareBallTestData(space)


@pytest.mark.smoke
class TestPoincareBall2Metric(
    PoincareBallMetricTestCase, metaclass=DataBasedParametrizer
):
    space = PoincareBall(dim=2, equip=False)
    space.equip_with_metric(PoincareBallMetric)
    testing_data = PoincareBall2MetricTestData()


class TestPoincareBallMetric(
    PoincareBallMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PoincareBall(dim=random.randint(2, 5))
    testing_data = PoincareBallMetricTestData(space)


class TestPoincareBallRetraction(
    RetractionTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PoincareBall(dim=random.randint(2, 5))
    testing_data = RetractionTestData(
        space,
        excluded_methods=(
            "lifting",
            "lifting_is_tangent",
            "lifting_after_retraction",
            "retraction_after_lifting",
        ),
    )
