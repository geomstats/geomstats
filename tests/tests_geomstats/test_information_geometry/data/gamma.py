import pytest

import geomstats.backend as gs
from geomstats.test.data import TestData

from ...test_geometry.data.base import VectorSpaceOpenSetTestData
from ...test_geometry.data.diffeo import DiffeoTestData
from ...test_geometry.data.riemannian_metric import RiemannianMetricTestData
from .base import InformationManifoldMixinTestData


class NaturalToStandardDiffeoTestData(DiffeoTestData):
    def diffeomorphism_test_data(self):
        data = [
            dict(base_point=gs.array([1.0, 1.0]), expected=gs.array([1.0, 1.0])),
            dict(base_point=gs.array([1.0, 2.0]), expected=gs.array([1.0, 0.5])),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))

    def inverse_test_data(self):
        data = [
            dict(image_point=gs.array([1.0, 1.0]), expected=gs.array([1.0, 1.0])),
            dict(image_point=gs.array([1.0, 2.0]), expected=gs.array([1.0, 0.5])),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))

    def tangent_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([2.0, 1.0]),
                base_point=gs.array([1.0, 2.0]),
                expected=gs.array([2.0, 0.75]),
            ),
            dict(
                tangent_vec=gs.array([1.0, 1.0]),
                base_point=gs.array([1.0, 1.0]),
                expected=gs.array([1.0, 0]),
            ),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))

    def inverse_tangent_test_data(self):
        data = [
            dict(
                image_tangent_vec=gs.array([2.0, 1.0]),
                image_point=gs.array([1.0, 2.0]),
                expected=gs.array([2.0, 0.75]),
            ),
            dict(
                image_tangent_vec=gs.array([1.0, 1.0]),
                image_point=gs.array([1.0, 1.0]),
                expected=gs.array([1.0, 0]),
            ),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))


class GammaDistributionsTestData(
    InformationManifoldMixinTestData, VectorSpaceOpenSetTestData
):
    fail_for_not_implemented_errors = False

    def point_to_pdf_against_scipy_test_data(self):
        return self.generate_random_data_with_samples()


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
    trials = 3
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {
        "dist_is_symmetric": {"atol": 1e-1},
        "squared_dist_is_symmetric": {"atol": 1e-1},
        "log_after_exp": {"atol": 1e-1},
        "exp_after_log": {"atol": 1e-1, "rtol": 1e-2},
    }
    xfails = ("exp_after_log",)

    def jacobian_christoffels_vec_test_data(self):
        return self.generate_vec_data()

    def scalar_curvature_against_closed_form_test_data(self):
        return self.generate_random_data()
