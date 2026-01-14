import random

import pytest

# from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.mds import MDS
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.mds import (
    MDSTestCase,
)

from .data.mds import MDSTestData  # MDSEuclideanTestData, MDSSPDTestData,


@pytest.fixture(
    scope="class",
    params=[
        Hypersphere(dim=random.randint(3, 4)),
        # SpecialOrthogonal(n=3, point_type="vector"),
        # SpecialOrthogonal(n=3, point_type="matrix"),
        SPDMatrices(3),
        Hyperboloid(dim=3),
    ],
)
def estimators(request):
    space = request.param
    request.cls.estimator = MDS(space)


@pytest.mark.usefixtures("estimators")
class TestMDS(MDSTestCase, metaclass=DataBasedParametrizer):
    testing_data = MDSTestData()


# class TestMDSEuclidean(MDSTestCase, metaclass=DataBasedParametrizer):
#     n = random.randint(2, 5)
#     estimator = MDS(Euclidean(dim=n), n_components=random.randint(2, 3))
#     testing_data = MDSEuclideanTestData(n)


# class TestMDSSPD(MDSTestCase, metaclass=DataBasedParametrizer):
#     n = random.randint(2, 5)
#     estimator = MDS(SPDMatrices(n, equip=True), n_components=random.randint(2, 3))
#     testing_data = MDSSPDTestData(n)
