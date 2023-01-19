import random

import pytest

from geomstats.geometry.positive_reals import PositiveReals
from geomstats.test.geometry.positive_reals import PositiveRealsTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.positive_reals_data import PositiveRealsTestData


class TestPositiveReals(PositiveRealsTestCase, metaclass=DataBasedParametrizer):
    space = PositiveReals()
    testing_data = PositiveRealsTestData()
