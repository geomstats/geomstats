import random

from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.poincare_polydisk import PoincarePolydisk
from geomstats.test_cases.geometry.nfold_manifold import NFoldManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.nfold_manifold import NFoldManifoldTestData
from .data.riemannian_metric import RiemannianMetricTestData


class TestPoincarePolydisk(
    NFoldManifoldTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PoincarePolydisk(n_disks=random.randint(2, 4), equip=False)
    testing_data = NFoldManifoldTestData(
        space, method_defaults={"projection_belongs": {"atol": 1e-6}}
    )


class TestPoincarePolydiskMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PoincarePolydisk(n_disks=random.randint(2, 4))
    testing_data = RiemannianMetricTestData(space)
