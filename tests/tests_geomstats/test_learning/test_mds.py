import random

import pytest

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.stratified.bhv_space import TreeSpace
from geomstats.learning.mds import MDS
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.mds import (
    EyePairwiseDistsTestCase,
    MDSTestCase,
    PairwiseDistsTestCase,
)

from .data.mds import (  # MDSEuclideanTestData, MDSSPDTestData,
    EyePairwiseDistsTestData,
    MDSTestData,
    PairwiseDistsTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        Hypersphere(dim=random.randint(3, 4)),
        SpecialOrthogonal(n=3, point_type="vector"),
        SpecialOrthogonal(n=3, point_type="matrix"),
        SPDMatrices(3),
        Hyperboloid(dim=3),
        TreeSpace(n_labels=random.randint(5, 8)),
    ],
)
def estimators(request):
    space = request.param
    request.cls.estimator = MDS(space)


@pytest.mark.usefixtures("estimators")
class TestPairwiseDists(PairwiseDistsTestCase, metaclass=DataBasedParametrizer):
    testing_data = PairwiseDistsTestData()


class TestEyePairwiseDists(EyePairwiseDistsTestCase, metaclass=DataBasedParametrizer):
    dim = random.randint(2, 5)
    point_mag = random.randint(1, 5)
    estimator = MDS(Euclidean(dim=dim), n_components=random.randint(2, 3))
    testing_data = EyePairwiseDistsTestData(dim=dim, n=point_mag)


@pytest.mark.usefixtures("estimators")
class TestMDS(MDSTestCase, metaclass=DataBasedParametrizer):
    testing_data = MDSTestData()
