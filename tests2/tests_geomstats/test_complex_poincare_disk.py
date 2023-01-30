import random

import pytest

from geomstats.geometry.complex_poincare_disk import ComplexPoincareDisk
from geomstats.test.geometry.complex_poincare_disk import ComplexPoincareDiskTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.complex_poincare_disk_data import ComplexPoincareDiskTestData


class TestComplexPoincareDisk(
    ComplexPoincareDiskTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ComplexPoincareDiskTestData()
    space = ComplexPoincareDisk()
