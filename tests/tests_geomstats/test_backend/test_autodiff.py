"""Unit tests for automatic differentiation in different backends."""

from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

import geomstats.backend as gs
from geomstats.test.test_case import autodiff_only, deselect_unless_np, torch_only
from geomstats.test_cases.backend.autodiff import (
    AutodiffTestCase,
    NotDifferentiableTestCase,
    NumpyRaisesTestCase,
)

from .data.autodiff import (
    AutodiffTestData,
    CustomGradientTestData,
    NotDifferentiableTestData,
    NumpyRaisesTestData,
)


@deselect_unless_np
class TestNumpyRaises(NumpyRaisesTestCase, metaclass=DataBasedParametrizer):
    dummy_func = lambda v: gs.sum(v**2)
    testing_data = NumpyRaisesTestData()


@autodiff_only
class TestAutodiff(AutodiffTestCase, metaclass=GeometricDataBasedParametrizer):
    testing_data = AutodiffTestData()


@autodiff_only
class TestCustomGradient(
    AutodiffTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = CustomGradientTestData()


@torch_only
class TestNotDifferentiable(
    NotDifferentiableTestCase, metaclass=DataBasedParametrizer
):
    testing_data = NotDifferentiableTestData()
