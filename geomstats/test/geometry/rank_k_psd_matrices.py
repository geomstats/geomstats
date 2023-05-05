from geomstats.geometry.general_linear import SquareMatrices
from geomstats.test.geometry.base import (
    FiberBundleTestCase,
    ManifoldTestCase,
    _ProjectionTestCaseMixins,
)
from geomstats.test.geometry.quotient_metric import QuotientMetricTestCase


class RankKPSDMatricesTestCase(_ProjectionTestCaseMixins, ManifoldTestCase):
    def _get_point_to_project(self, n_points):
        return SquareMatrices(self.space.n).random_point(n_points)


class BuresWassersteinBundleTestCase(FiberBundleTestCase):
    pass


class PSDBuresWassersteinMetricTestCase(QuotientMetricTestCase):
    pass
