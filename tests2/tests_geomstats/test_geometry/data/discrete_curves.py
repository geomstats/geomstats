from .base import LevelSetTestData, _ProjectionMixinsTestData
from .fiber_bundle import FiberBundleTestData
from .manifold import ManifoldTestData
from .quotient_metric import QuotientMetricTestData


class DiscreteCurvesTestData(_ProjectionMixinsTestData, ManifoldTestData):
    pass


class SRVShapeBundleTestData(FiberBundleTestData):
    fail_for_not_implemented_errors = False
    skips = (
        "horizontal_geodesic_vec",
        "align_vec",
        "log_after_align_is_horizontal",
    )
    xfails = (
        # not robust
        "horizontal_geodesic_has_horizontal_derivative",
        "tangent_riemannian_submersion_after_horizontal_lift",
    )
    tolerances = {
        "tangent_vector_projections_orthogonality_with_metric": {"atol": 5e-1},
        "vertical_projection_is_vertical": {"atol": 1e-1},
        "horizontal_projection_is_horizontal": {"atol": 1e-1},
        "horizontal_geodesic_has_horizontal_derivative": {"atol": 1e-1},
        "horizontal_lift_is_horizontal": {"atol": 1e-1},
        "tangent_riemannian_submersion_after_vertical_projection": {"atol": 5e-1},
        "tangent_riemannian_submersion_after_horizontal_lift": {"atol": 5e-1},
    }

    def tangent_vector_projections_orthogonality_with_metric_test_data(self):
        return self.generate_random_data()

    def horizontal_geodesic_vec_test_data(self):
        data = []
        for n_times in [20]:
            data.extend(
                [dict(n_reps=n_reps, n_times=n_times) for n_reps in self.N_VEC_REPS]
            )
        return self.generate_tests(data)

    def horizontal_geodesic_has_horizontal_derivative_test_data(self):
        data = []
        for n_times in [20]:
            data.extend([dict(n_points=n_points, n_times=n_times) for n_points in [1]])
        return self.generate_tests(data)


class ClosedDiscreteCurvesTestData(LevelSetTestData):
    fail_for_not_implemented_errors = False

    skips = ("srv_projection_vec",)

    def srv_projection_vec_test_data(self):
        return self.generate_vec_data()

    def projection_is_itself_test_data(self):
        return self.generate_random_data()

    def random_point_is_closed_test_data(self):
        return self.generate_random_data()


class SRVQuotientMetricTestData(QuotientMetricTestData):
    fail_for_not_implemented_errors = False
