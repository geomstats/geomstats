from .riemannian_metric import (
    RiemannianMetricComparisonTestData,
    RiemannianMetricTestData,
)


class QuotientMetricTestData(RiemannianMetricTestData):
    pass


class SPDBuresWassersteinQuotientMetricCmpTestData(RiemannianMetricComparisonTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {"geodesic_bvp_random": {"atol": 1e-3}, "log_random": {"atol": 1e-3}}

    skips = (
        "parallel_transport_ivp_random",
        "parallel_transport_bvp_random",
    )
