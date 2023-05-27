from tests2.tests_geomstats.test_geometry.data.base import OpenSetTestData
from tests2.tests_geomstats.test_geometry.data.riemannian_metric import (
    RiemannianMetricTestData,
)
from tests2.tests_geomstats.test_information_geometry.data.base import (
    InformationManifoldMixinTestData,
)


class GeometricDistributionsTestData(InformationManifoldMixinTestData, OpenSetTestData):
    pass


class GeometricMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    xfails = (
        # rarely fails
        "log_after_exp",
    )
