import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.complex_poincare_disk import ComplexPoincareDisk
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
    RiemannianMetricTestCase,
)
from geomstats.variants.geometry.complex_poincare_disk import (
    PullbackComplexPoincareDiskMetric,
)

from .data.complex_poincare_disk import (
    ComplexPoincareDiskMetricTestData,
    ComplexPoincareDiskTestData,
)
from .data.riemannian_metric import RiemannianMetricComparisonTestData


class TestComplexPoincareDisk(
    VectorSpaceOpenSetTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ComplexPoincareDisk(equip=False)
    testing_data = ComplexPoincareDiskTestData(space)


class TestComplexPoincareDiskMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ComplexPoincareDisk()
    testing_data = ComplexPoincareDiskMetricTestData(space)


@pytest.mark.redundant
class TestPullbackComplexPoincareDiskMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ComplexPoincareDisk()
    other_space = ComplexPoincareDisk(equip=False).equip_with_metric(
        PullbackComplexPoincareDiskMetric
    )
    testing_data = RiemannianMetricComparisonTestData(space)
