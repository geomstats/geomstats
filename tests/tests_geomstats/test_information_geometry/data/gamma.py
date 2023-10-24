import geomstats.backend as gs
from geomstats.test.data import TestData

from ...test_geometry.data.base import OpenSetTestData
from ...test_geometry.data.riemannian_metric import RiemannianMetricTestData
from .base import InformationManifoldMixinTestData


class GammaDistributionsTestData(InformationManifoldMixinTestData, OpenSetTestData):
    fail_for_not_implemented_errors = False

    def point_to_pdf_against_scipy_test_data(self):
        return self.generate_random_data_with_samples()

    def natural_to_standard_vec_test_data(self):
        return self.generate_vec_data()

    def standard_to_natural_vec_test_data(self):
        return self.generate_vec_data()

    def standard_to_natural_after_natural_to_standard_test_data(self):
        return self.generate_random_data()

    def tangent_natural_to_standard_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_standard_to_natural_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_standard_to_natural_after_tangent_natural_to_standard_test_data(self):
        return self.generate_random_data()


class GammaDistributionsSmokeTestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([0.1, -1.0]), expected=False),
            dict(point=gs.array([0.1, 1.0]), expected=True),
            dict(point=gs.array([0.0, 1.0, 0.3]), expected=False),
            dict(point=gs.array([-1.0, 0.3]), expected=False),
            dict(point=gs.array([0.1, 5]), expected=True),
        ]
        return self.generate_tests(data)

    def natural_to_standard_test_data(self):
        data = [
            dict(point=gs.array([1.0, 1.0]), expected=gs.array([1.0, 1.0])),
            dict(point=gs.array([1.0, 2.0]), expected=gs.array([1.0, 0.5])),
        ]
        return self.generate_tests(data)

    def standard_to_natural_test_data(self):
        data = [
            dict(point=gs.array([1.0, 1.0]), expected=gs.array([1.0, 1.0])),
            dict(point=gs.array([1.0, 2.0]), expected=gs.array([1.0, 0.5])),
        ]
        return self.generate_tests(data)

    def tangent_natural_to_standard_test_data(self):
        data = [
            dict(
                vec=gs.array([2.0, 1.0]),
                base_point=gs.array([1.0, 2.0]),
                expected=gs.array([2.0, 0.75]),
            ),
            dict(
                vec=gs.array([1.0, 1.0]),
                base_point=gs.array([1.0, 1.0]),
                expected=gs.array([1.0, 0]),
            ),
        ]
        return self.generate_tests(data)

    def tangent_standard_to_natural_test_data(self):
        data = [
            dict(
                vec=gs.array([2.0, 1.0]),
                base_point=gs.array([1.0, 2.0]),
                expected=gs.array([2.0, 0.75]),
            ),
            dict(
                vec=gs.array([1.0, 1.0]),
                base_point=gs.array([1.0, 1.0]),
                expected=gs.array([1.0, 0]),
            ),
        ]
        return self.generate_tests(data)

    def maximum_likelihood_fit_test_data(self):
        smoke_data = [
            dict(
                data=gs.array([1, 2, 3, 4]),
                expected=gs.array([4.26542805, 2.5]),
            ),
            dict(
                data=[[1, 2, 3, 4, 5], [1, 2, 3, 4]],
                expected=gs.array([[3.70164381, 3.0], [4.26542805, 2.5]]),
            ),
            dict(
                data=[gs.array([1, 2, 3, 4, 5]), gs.array([1, 2, 3, 4])],
                expected=gs.array([[3.70164381, 3.0], [4.26542805, 2.5]]),
            ),
            dict(
                data=[[1, 2, 3, 4]],
                expected=gs.array([4.26542805, 2.5]),
            ),
            dict(
                data=gs.array([1, 2, 3, 4]),
                expected=gs.array([4.26542805, 2.5]),
            ),
            dict(
                data=[gs.array([1, 2, 3, 4])],
                expected=gs.array([4.26542805, 2.5]),
            ),
            dict(
                data=[[1, 2, 3, 4, 5], gs.array([1, 2, 3, 4])],
                expected=gs.array([[3.70164381, 3.0], [4.26542805, 2.5]]),
            ),
        ]
        return self.generate_tests(smoke_data)


class GammaMetricTestData(RiemannianMetricTestData):
    trials = 5
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {
        "dist_is_symmetric": {"atol": 1e-3},
        "log_after_exp": {"atol": 1e-2},
        "exp_after_log": {"atol": 1e-2},
        "squared_dist_is_symmetric": {"atol": 1e-3},
    }
    xfails = tuple(tolerances.keys())

    def jacobian_christoffels_vec_test_data(self):
        return self.generate_vec_data()

    def scalar_curvature_against_closed_form_test_data(self):
        return self.generate_random_data()
