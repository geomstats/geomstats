from geomstats.geometry.complex_poincare_disk import ComplexPoincareDisk
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import np_only
from geomstats.test_cases.geometry.base import ComplexVectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.complex_riemannian_metric import (
    ComplexRiemannianMetricTestCase,
)

from .data.complex_poincare_disk import (
    ComplexPoincareDiskMetricTestData,
    ComplexPoincareDiskTestData,
)


class TestComplexPoincareDisk(
    ComplexVectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer
):
    space = ComplexPoincareDisk(equip=False)
    testing_data = ComplexPoincareDiskTestData()


@np_only
class TestComplexPoincareDiskMetric(
    ComplexRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = ComplexPoincareDisk()
    testing_data = ComplexPoincareDiskMetricTestData()
