import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.positive_hypersphere import PositiveHypersphere
from geomstats.geometry.simplex import OpenSimplex, SqrtDiffeo
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase

from .data.base import LevelSetTestData
from .data.diffeo import DiffeoTestData
from .data.simplex import OpenSimplex3TestData


class TestSqrtDiffeo(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    _dim = random.randint(2, 5)
    domain_space = OpenSimplex(dim=_dim)
    image_space = PositiveHypersphere(dim=_dim, equip=False)
    diffeo = SqrtDiffeo()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestOpenSimplex(LevelSetTestCase, metaclass=GeometricDataBasedParametrizer):
    space = OpenSimplex(dim=random.randint(2, 5))
    testing_data = LevelSetTestData(space)


@pytest.mark.smoke
class TestOpenSimplex3(LevelSetTestCase, metaclass=DataBasedParametrizer):
    space = OpenSimplex(dim=3)
    testing_data = OpenSimplex3TestData()
