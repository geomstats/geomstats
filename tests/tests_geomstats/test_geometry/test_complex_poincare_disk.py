from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.complex_poincare_disk import ComplexPoincareDisk
from geomstats.test_cases.geometry.base import ComplexVectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.complex_riemannian_metric import (
    ComplexRiemannianMetricTestCase,
)

from .data.complex_poincare_disk import (
    ComplexPoincareDiskMetricTestData,
    ComplexPoincareDiskTestData,
)


class TestComplexPoincareDisk(
    ComplexVectorSpaceOpenSetTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ComplexPoincareDisk(equip=False)
    testing_data = ComplexPoincareDiskTestData(space)


class TestComplexPoincareDiskMetric(
    ComplexRiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ComplexPoincareDisk()
    testing_data = ComplexPoincareDiskMetricTestData(space)
