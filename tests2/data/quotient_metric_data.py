from tests2.data.base_data import RiemannianMetricTestData
from tests2.data.comparison_data import RiemannianMetricComparisonTestData


class QuotientMetricTestData(RiemannianMetricTestData):
    pass


class SPDBuresWassersteinQuotientMetricTestData(RiemannianMetricComparisonTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {"geodesic_bvp": {"atol": 1e-6}, "log": {"atol": 1e-6}}
