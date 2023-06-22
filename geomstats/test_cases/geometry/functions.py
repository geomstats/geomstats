from geomstats.test_cases.geometry.base import _ProjectionTestCaseMixins
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class HilbertSphereTestCase(_ProjectionTestCaseMixins, ManifoldTestCase):
    pass


class HilbertSphereMetricTestCase(RiemannianMetricTestCase):
    pass
