import pytest

from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.radial_kernel_functions import (
    RadialKernelFunctionsTestCase,
)

from .data.radial_kernel_functions import RadialKernelFunctionsTestData


@pytest.mark.smoke
class TestRadialKernelFunctions(
    RadialKernelFunctionsTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = RadialKernelFunctionsTestData()
