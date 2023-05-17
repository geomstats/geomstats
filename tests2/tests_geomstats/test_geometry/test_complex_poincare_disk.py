import random

import pytest

from geomstats.geometry.complex_poincare_disk import ComplexPoincareDisk
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.complex_poincare_disk import (
    ComplexPoincareDiskTestCase,
)

from .data.complex_poincare_disk import ComplexPoincareDiskTestData


class TestComplexPoincareDisk(
    ComplexPoincareDiskTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ComplexPoincareDiskTestData()
    space = ComplexPoincareDisk()
