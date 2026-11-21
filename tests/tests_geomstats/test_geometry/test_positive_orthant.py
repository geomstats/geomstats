import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.positive_orthant import PositiveOrthant, ScaledReciprocalDiffeo
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase

from .data.base import VectorSpaceOpenSetTestData
from .data.positive_orthant import (
    PositiveOrthant2TestData,
    ScaledReciprocalDiffeoTestData,
)


class TestScaledReciprocalDiffeo(
    DiffeoTestCase, metaclass=GeometricDataBasedParametrizer
):
    domain_space = PositiveOrthant(dim=2, equip=False)
    image_space = PositiveOrthant(dim=2, equip=False)
    diffeo = ScaledReciprocalDiffeo()

    testing_data = ScaledReciprocalDiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestPositiveOrthant(
    VectorSpaceOpenSetTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PositiveOrthant(dim=random.randint(2, 5), equip=False)
    testing_data = VectorSpaceOpenSetTestData(space)


@pytest.mark.smoke
class TestPositiveOrthant2(VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer):
    space = PositiveOrthant(dim=2, equip=False)
    testing_data = PositiveOrthant2TestData()
