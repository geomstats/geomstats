"""Test Principal Nested Spheres."""

import pytest

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.principal_nested_spheres import PrincipalNestedSpheres
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import np_and_autograd_only
from geomstats.test_cases.learning.principal_nested_spheres import (
    PrincipalNestedSpheresTestCase,
)

from .data.principal_nested_spheres import PrincipalNestedSpheresTestData


@pytest.fixture(
    scope="class",
    params=[
        Hypersphere(dim=2),
        Hypersphere(dim=3),
    ],
)
def estimators(request):
    """Fixture for PrincipalNestedSpheres estimators with different sphere dimensions."""
    space = request.param
    request.cls.estimator = PrincipalNestedSpheres(
        space=space,
        n_init=3,  # Reduce for faster testing
        max_iter=100,
        tol=1e-6,
    )


@np_and_autograd_only
@pytest.mark.usefixtures("estimators")
class TestPrincipalNestedSpheres(
    PrincipalNestedSpheresTestCase,
    metaclass=DataBasedParametrizer,
):
    """Test class for Principal Nested Spheres using parametrized test data."""

    testing_data = PrincipalNestedSpheresTestData()
