import random

import pytest

from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.stratified.bhv_space import TreeSpace
from geomstats.learning.mds import MDS
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.mds import (
    MDSTestCase,
    PairwiseDistsTestCase,
)

from .data.mds import (
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
def spaces(request):
    space = request.param
    request.cls.space = space


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


@pytest.mark.usefixtures("spaces")
class TestPairwiseDists(PairwiseDistsTestCase, metaclass=DataBasedParametrizer):
    testing_data = PairwiseDistsTestData()


@pytest.mark.usefixtures("estimators")
class TestMDS(MDSTestCase, metaclass=DataBasedParametrizer):
    testing_data = MDSTestData()
