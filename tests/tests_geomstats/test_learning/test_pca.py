import pytest

from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.pca import TangentPCA
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import np_and_autograd_only
from geomstats.test_cases.learning.pca import TangentPCATestCase

from .data.pca import TangentPCATestData


@pytest.fixture(
    scope="class",
    params=[
        SpecialOrthogonal(n=3, point_type="vector", equip=False),
        SPDMatrices(3),
        SpecialEuclidean(n=3, equip=False),
    ],
)
def estimators(request):
    space = request.param
    request.cls.estimator = TangentPCA(space)


@np_and_autograd_only
@pytest.mark.usefixtures("estimators")
class TestTangentPCA(
    TangentPCATestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = TangentPCATestData()
