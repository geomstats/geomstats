import geomstats.backend as gs
from geomstats.test.data import TestData

from ...test_geometry.data.base import VectorSpaceOpenSetTestData
from ...test_geometry.data.riemannian_metric import RiemannianMetricTestData
from .base import InformationManifoldMixinTestData


class DirichletDistributionsTestData(
    InformationManifoldMixinTestData, VectorSpaceOpenSetTestData
):
    fail_for_not_implemented_errors = False


class DirichletDistributions3TestData(TestData):
    def belongs_test_data(self):
        smoke_data = [
            dict(point=gs.array([0.1, 1.0, 0.3]), expected=True),
            dict(point=gs.array([0.1, 1.001]), expected=False),
            dict(point=gs.array([-0.001, 1.0, 0.3]), expected=False),
        ]
        return self.generate_tests(smoke_data)


class DirichletMetricTestData(RiemannianMetricTestData):
    trials = 3
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {
        "dist_is_symmetric": {"atol": 1e-2},
        "log_after_exp": {"atol": 1e-3},
        "exp_after_log": {"atol": 1e-2},
        "squared_dist_is_symmetric": {"atol": 1e-1},
    }

    xfails = (
        "dist_is_symmetric",
        "squared_dist_is_symmetric",
        "exp_after_log",
        "exp_diagonal_is_totally_geodesic",
    )

    def jacobian_christoffels_vec_test_data(self):
        return self.generate_vec_data()

    def sectional_curvature_is_negative_test_data(self):
        return self.generate_random_data()

    def exp_diagonal_is_totally_geodesic_test_data(self):
        data = [dict(param=gs.random.rand(1), tangent_param=gs.random.rand(1))]
        return self.generate_tests(data)
