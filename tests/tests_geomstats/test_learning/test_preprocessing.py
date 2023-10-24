import pytest

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.preprocessing import ToTangentSpace
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.preprocessing import ToTangentSpaceTestCase

from .data.preprocessing import ToTangentSpaceNdim2TestData, ToTangentSpaceTestData

# TODO: fix main code and make this unique


@pytest.fixture(
    scope="class",
    params=[
        Hypersphere(dim=4),
        Hyperboloid(dim=3),
        Euclidean(dim=2),
        Minkowski(dim=2),
        SpecialOrthogonal(n=3, point_type="vector", equip=False),
    ],
)
def estimators(request):
    space = request.param
    request.cls.estimator = ToTangentSpace(space)


@pytest.mark.usefixtures("estimators")
class TestToTangent(ToTangentSpaceTestCase, metaclass=DataBasedParametrizer):
    testing_data = ToTangentSpaceTestData()


@pytest.fixture(
    scope="class",
    params=[
        SpecialOrthogonal(n=3, point_type="matrix", equip=False),
        SPDMatrices(3),
    ],
)
def estimators_ndim2(request):
    space = request.param
    request.cls.estimator = ToTangentSpace(space)


@pytest.mark.usefixtures("estimators_ndim2")
class TestToTangentNdim2(ToTangentSpaceTestCase, metaclass=DataBasedParametrizer):
    testing_data = ToTangentSpaceNdim2TestData()
