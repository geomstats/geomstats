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
    other_space.equip_with_metric(ScalarProductMetric(other_space, scale))

    testing_data = NFoldMetricScalesTestData()


def test_pow_constructor():
    nfold_1 = SpecialOrthogonal(n=2) ** 50
    assert isinstance(nfold_1, NFoldManifold)

    nfold_2 = Euclidean(dim=3) ** 3
    assert isinstance(nfold_2, NFoldManifold)

    nfold_3 = SpecialOrthogonal(n=5) ** 99
    assert isinstance(nfold_3, NFoldManifold)


def test_pow_constructor_vec():
    nfold = SpecialOrthogonal(2) ** 50

    one_point = nfold.random_point()
    assert one_point.shape[0] == nfold.base_manifold.dim * nfold.n_copies

    mult_points = nfold.random_point(n_samples=5)
    assert mult_points.shape[0] == 5
    assert mult_points.shape[1] == nfold.base_manifold.dim * nfold.n_copies


def test_pow_invalid_nfold():
    with pytest.raises(ValueError):
        _ = SpecialOrthogonal(3) ** 0

    with pytest.raises(ValueError):
        _ = SpecialOrthogonal(3) ** -5
