import pytest
from polpo.testing.geometric import GeometricCaseData

import geomstats.backend as gs

from .dirichlet import DirichletDistributionsTestData, DirichletMetricTestData


class BetaDistributionsTestData(DirichletDistributionsTestData):
    def point_to_pdf_against_scipy_test_data(self):
        return self._generate_random_data_with_samples()

    @pytest.mark.smoke
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([0.1, 1.0, 0.3]), expected=False),
            dict(point=gs.array([0.1, 1.0]), expected=True),
            dict(point=gs.array([-1.0, 0.3]), expected=False),
        ]
        return data


class BetaMetricTestData(DirichletMetricTestData):
    def metric_det_vec_test_data(self):
        return self.generate_vectorization_data(arg_names="point", on_metric=True)

    def metric_det_against_metric_matrix_test_data(self):
        return self.generate_random_data(arg_names="point")

    def metric_det_lower_bound_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def sectional_curvature_lower_bound_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )

    @pytest.mark.smoke
    def metric_matrix_test_data(self):
        data = [
            dict(
                base_point=gs.array([1.0, 1.0]),
                expected=gs.array([[1.0, -0.644934066], [-0.644934066, 1.0]]),
            )
        ]
        return data


class BetaMetricAsDirichletCmpTestData(GeometricCaseData):
    def metric_matrix_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def christoffels_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def sectional_curvature_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )
