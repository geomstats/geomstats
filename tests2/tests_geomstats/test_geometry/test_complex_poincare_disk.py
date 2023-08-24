from geomstats.geometry.complex_poincare_disk import (
    ComplexPoincareDisk,
    ComplexPoincareDiskMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import ComplexOpenSetTestCase
from geomstats.test_cases.geometry.complex_riemannian_metric import (
    ComplexRiemannianMetricTestCase,
)

from .data.complex_poincare_disk import (
    ComplexPoincareDiskMetricTestData,
    ComplexPoincareDiskTestData,
)


class TestComplexPoincareDisk(ComplexOpenSetTestCase, metaclass=DataBasedParametrizer):
    space = ComplexPoincareDisk(equip=False)
    testing_data = ComplexPoincareDiskTestData()


class TestComplexPoincareDiskMetric(
    ComplexRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = ComplexPoincareDisk(equip=False)
    space.equip_with_metric(ComplexPoincareDiskMetric)
    testing_data = ComplexPoincareDiskMetricTestData()
