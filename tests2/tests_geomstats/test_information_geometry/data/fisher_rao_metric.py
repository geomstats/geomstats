from geomstats.test.data import TestData
from tests2.tests_geomstats.test_geometry.data.riemannian_metric import (
    RiemannianMetricComparisonTestData,
)


class FisherRaoMetricCmpTestData(TestData):
    # TODO: inherit all from RiemannianMetricComparisonTestData?
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    def metric_matrix_test_data(self):
        return self.generate_random_data()
