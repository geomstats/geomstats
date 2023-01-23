from geomstats.geometry.general_linear import SquareMatrices
from geomstats.test.geometry.base import (
    FiberBundleTestCase,
    ManifoldTestCase,
    _ProjectionTestCaseMixins,
)
from geomstats.test.geometry.full_rank_matrices import FullRankMatricesTestCase


class RankKPSDMatricesTestCase(_ProjectionTestCaseMixins, ManifoldTestCase):
    def _get_point_to_project(self, n_points):
        return SquareMatrices(self.space.n).random_point(n_points)


class BuresWassersteinBundleTestCase(FullRankMatricesTestCase, FiberBundleTestCase):
    pass
