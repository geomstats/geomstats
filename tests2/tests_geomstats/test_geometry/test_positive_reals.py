import random

import pytest

from geomstats.geometry.positive_reals import PositiveReals
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.positive_reals import PositiveRealsTestCase

from .data.positive_reals import PositiveRealsTestData


class TestPositiveReals(PositiveRealsTestCase, metaclass=DataBasedParametrizer):
    space = PositiveReals()
    testing_data = PositiveRealsTestData()
