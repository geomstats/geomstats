from tests2.tests_geomstats.test_geometry.data.manifold import ManifoldTestData
from tests2.tests_geomstats.test_geometry.data.riemannian_metric import (
    RiemannianMetricTestData,
)


class HilbertSphereTestData(ManifoldTestData):
    skips = ("not_belongs",)


class HilbertSphereMetricTestData(RiemannianMetricTestData):
    trials = 3
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False

    skips = (
        "exp_after_log",
        "exp_belongs",
        "geodesic_boundary_points",
        "geodesic_bvp_belongs",
        "geodesic_bvp_reverse",
        "geodesic_ivp_belongs",
        "log_after_exp",
        "log_is_tangent",
    )
