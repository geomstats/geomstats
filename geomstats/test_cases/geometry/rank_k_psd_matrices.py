from geomstats.test.random import RankKPSDMatricesRandomDataGenerator
from geomstats.test_cases.geometry.base import _ProjectionTestCaseMixins
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.quotient_metric import QuotientMetricTestCase


class RankKPSDMatricesTestCase(_ProjectionTestCaseMixins, ManifoldTestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RankKPSDMatricesRandomDataGenerator(self.space)


class BuresWassersteinBundleTestCase(FiberBundleTestCase):
    pass


class PSDBuresWassersteinMetricTestCase(QuotientMetricTestCase):
    pass
