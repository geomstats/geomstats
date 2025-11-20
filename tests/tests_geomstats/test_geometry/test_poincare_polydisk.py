import random

from geomstats.geometry.poincare_polydisk import PoincarePolydisk
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.nfold_manifold import (
    NFoldManifoldTestCase,
    NFoldMetricTestCase,
)

from .data.poincare_polydisk import (
    PoincarePolydiskMetricTestData,
    PoincarePolydiskTestData,
)


class TestPoincarePolydisk(NFoldManifoldTestCase, metaclass=DataBasedParametrizer):
    n_disks = random.randint(2, 4)
    space = PoincarePolydisk(n_disks=n_disks, equip=False)
    testing_data = PoincarePolydiskTestData()


class TestPoincarePolydiskMetric(NFoldMetricTestCase, metaclass=DataBasedParametrizer):
    n_disks = random.randint(2, 4)
    space = PoincarePolydisk(n_disks=n_disks)
    testing_data = PoincarePolydiskMetricTestData()
