import math
import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.open_interval import (
    ArcsinSqrtDiffeo,
    ArtanhSqrtDiffeo,
    OpenInterval,
)
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.euclidean import EuclideanMetricTestCase

from .data.base import VectorSpaceOpenSetTestData
from .data.diffeo import DiffeoTestData
from .data.euclidean import EuclideanMetricTestData
from .data.open_interval import (
    OpenInterval01MetricTestData,
    OpenInterval01TestData,
)


class TestArcsinSqrtDiffeo(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    _scale = random.uniform(1.0, 5.0)
    domain_space = OpenInterval(0.0, 1.0, equip=False)
    image_space = OpenInterval(0.0, math.pi * math.sqrt(_scale), equip=False)
    diffeo = ArcsinSqrtDiffeo(_scale)

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestArtanhSqrtDiffeo(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    domain_space = OpenInterval(0.0, 1.0, equip=False)
    image_space = OpenInterval(0.0, math.inf, equip=False)
    diffeo = ArtanhSqrtDiffeo()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


def _create_open_intervals(equip):
    lower_bound = random.uniform(-2.0, 2.0)
    upper_bound = lower_bound + random.uniform(0.5, 4.0)
    return [
        OpenInterval(lower_bound, upper_bound, equip=equip),
        OpenInterval(lower_bound, math.inf, equip=equip),
        OpenInterval(-math.inf, upper_bound, equip=equip),
    ]


@pytest.fixture(scope="class", params=_create_open_intervals(equip=False))
def spaces(request):
    space = request.cls.space = request.param
    request.cls.testing_data.space = space


@pytest.mark.usefixtures("spaces")
class TestOpenInterval(
    VectorSpaceOpenSetTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = VectorSpaceOpenSetTestData()


@pytest.mark.smoke
class TestOpenInterval01(VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer):
    space = OpenInterval(0.0, 1.0, equip=False)
    testing_data = OpenInterval01TestData()


@pytest.fixture(scope="class", params=_create_open_intervals(equip=True))
def equipped_spaces(request):
    space = request.cls.space = request.param
    request.cls.testing_data.space = space


@pytest.mark.usefixtures("equipped_spaces")
class TestOpenIntervalMetric(
    EuclideanMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = EuclideanMetricTestData(
        excluded_methods=("sectional_curvature",),
        generator_kwargs={"within_injectivity_radius": True},
    )


@pytest.mark.smoke
class TestOpenInterval01Metric(
    EuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    space = OpenInterval(0.0, 1.0)
    testing_data = OpenInterval01MetricTestData()
