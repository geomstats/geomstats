import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.nfold_manifold import NFoldManifold, NFoldMetric
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.nfold_manifold import (
    NFoldManifoldTestCase,
    NFoldMetricScalesTestCase,
)

from .data.nfold_manifold import NFoldManifoldSOTestData, NFoldMetricScalesTestData


@pytest.mark.smoke
class TestNFoldManifoldSO(NFoldManifoldTestCase, metaclass=DataBasedParametrizer):
    space = NFoldManifold(SpecialOrthogonal(3, equip=False), 2, equip=False)
    testing_data = NFoldManifoldSOTestData()


class TestNFoldMetricScales(NFoldMetricScalesTestCase, metaclass=DataBasedParametrizer):
    scale = 2.0
    space = NFoldManifold(Euclidean(dim=3), n_copies=1, equip=False)
    space.equip_with_metric(NFoldMetric, scales=gs.array([scale]))

    other_space = Euclidean(dim=3)
    other_space.metric = ScalarProductMetric(other_space.metric, scale)

    testing_data = NFoldMetricScalesTestData()
