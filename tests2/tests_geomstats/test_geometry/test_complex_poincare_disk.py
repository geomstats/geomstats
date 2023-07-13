from geomstats.geometry.complex_poincare_disk import ComplexPoincareDisk
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import ComplexOpenSetTestCase

from .data.complex_poincare_disk import ComplexPoincareDiskTestData


class TestComplexPoincareDisk(ComplexOpenSetTestCase, metaclass=DataBasedParametrizer):
    testing_data = ComplexPoincareDiskTestData()
    space = ComplexPoincareDisk()
