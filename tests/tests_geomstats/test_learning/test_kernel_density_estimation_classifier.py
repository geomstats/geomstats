import pytest

from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.kernel_density_estimation_classifier import (
    KernelDensityEstimationClassifierTestCase,
)

from .data.kernel_density_estimation_classifier import (
    KernelDensityEstimationClassifierTestData,
)


@pytest.mark.smoke
class TestKernelDensityEstimationClassifier(
    KernelDensityEstimationClassifierTestCase, metaclass=DataBasedParametrizer
):
    testing_data = KernelDensityEstimationClassifierTestData()
