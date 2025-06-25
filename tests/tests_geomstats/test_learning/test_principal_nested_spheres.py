import pytest

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.principal_nested_spheres import PrincipalNestedSpheres
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import np_and_autograd_only
from geomstats.test_cases.learning.principal_nested_spheres import PrincipalNestedSpheresTestCase

from tests_geomstats.test_learning.data.principal_nested_spheres import PrincipalNestedSpheresTestData


@pytest.fixture(
    scope="class",
    params=[Hypersphere(dim=2), Hypersphere(dim=3)],
)
def spheres(request):
    """
    Fixture for different sphere dimensions.
    """
    sphere = request.param
    # Attach estimator instance to the test class
    request.cls.estimator = PrincipalNestedSpheres(sphere=sphere)
    return sphere


@np_and_autograd_only
@pytest.mark.usefixtures("spheres")
class TestPrincipalNestedSpheres(
    PrincipalNestedSpheresTestCase,
    metaclass=DataBasedParametrizer,
):
    """
    Test suite for PrincipalNestedSpheres algorithm using data-driven parametrization.
    """
    testing_data = PrincipalNestedSpheresTestData()
