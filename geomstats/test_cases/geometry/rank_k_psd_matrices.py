from geomstats.test.random import RankKPSDMatricesRandomDataGenerator
from geomstats.test_cases.geometry.base import _ProjectionTestCaseMixins
from geomstats.test_cases.geometry.manifold import ManifoldTestCase


class RankKPSDMatricesTestCase(_ProjectionTestCaseMixins, ManifoldTestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RankKPSDMatricesRandomDataGenerator(self.space)
