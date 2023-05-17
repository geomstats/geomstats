from geomstats.geometry.general_linear import SquareMatrices
from geomstats.test_cases.geometry.base import _ProjectionTestCaseMixins
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.quotient_metric import QuotientMetricTestCase


class RankKPSDMatricesTestCase(_ProjectionTestCaseMixins, ManifoldTestCase):
    def _get_point_to_project(self, n_points):
        return SquareMatrices(self.space.n).random_point(n_points)


class BuresWassersteinBundleTestCase(FiberBundleTestCase):
    pass


class PSDBuresWassersteinMetricTestCase(QuotientMetricTestCase):
    pass
