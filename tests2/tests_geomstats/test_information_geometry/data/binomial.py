from tests2.tests_geomstats.test_geometry.data.base import OpenSetTestData
from tests2.tests_geomstats.test_geometry.data.riemannian_metric import (
    RiemannianMetricTestData,
)
from tests2.tests_geomstats.test_information_geometry.data.base import (
    InformationManifoldMixinTestData,
)


class BinomialDistributionsTestData(InformationManifoldMixinTestData, OpenSetTestData):
    pass


class BinomialMetricTestData(RiemannianMetricTestData):
    trials = 2
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
