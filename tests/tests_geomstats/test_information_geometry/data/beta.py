import geomstats.backend as gs
from geomstats.test.data import TestData

from .dirichlet import DirichletDistributionsTestData, DirichletMetricTestData


class BetaDistributionsTestData(DirichletDistributionsTestData):
    def point_to_pdf_against_scipy_test_data(self):
        return self.generate_random_data_with_samples()


class BetaDistributionsSmokeTestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([0.1, 1.0, 0.3]), expected=False),
            dict(point=gs.array([0.1, 1.0]), expected=True),
            dict(point=gs.array([-1.0, 0.3]), expected=False),
        ]
        return self.generate_tests(data)


class BetaMetricTestData(DirichletMetricTestData):
    def metric_det_vec_test_data(self):
        return self.generate_vec_data()

    def metric_det_against_metric_matrix_test_data(self):
        return self.generate_random_data()

    def metric_det_lower_bound_test_data(self):
        return self.generate_random_data()

    def sectional_curvature_against_closed_form_test_data(self):
        return self.generate_random_data()

    def sectional_curvature_lower_bound_test_data(self):
        return self.generate_random_data()

    def metric_matrix_against_closed_form_test_data(self):
        return self.generate_random_data()

    def christoffels_against_closed_form_test_data(self):
        return self.generate_random_data()


class BetaMetricSmokeTestData(TestData):
    def metric_matrix_test_data(self):
        data = [
            dict(
                base_point=gs.array([1.0, 1.0]),
                expected=gs.array([[1.0, -0.644934066], [-0.644934066, 1.0]]),
            )
        ]
        return self.generate_tests(data)
