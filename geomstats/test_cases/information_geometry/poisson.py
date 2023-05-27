from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)


class PoissonDistributionsTestCase(InformationManifoldMixinTestCase, OpenSetTestCase):
    pass


class PoissonMetricTestCase(RiemannianMetricTestCase):
    pass
