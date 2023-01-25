from tests2.data.base_data import (
    FiberBundleTestData,
    ManifoldTestData,
    _ProjectionMixinsTestData,
)


class DiscreteCurvesTestData(_ProjectionMixinsTestData, ManifoldTestData):
    pass


class SRVShapeBundleTestData(DiscreteCurvesTestData, FiberBundleTestData):
    skips = (
        "horizontal_geodesic_vec",
        "align_vec",
        "log_after_align_is_horizontal",
        # due to lack of base
        "lift_vec",
        "lift_belongs_to_total_space",
        "riemannian_submersion_after_lift",
        "riemannian_submersion_belongs_to_base",
        "tangent_riemannian_submersion_is_tangent",
        "tangent_riemannian_submersion_after_horizontal_lift",
        # not implemented
        "integrability_tensor_vec",
        "integrability_tensor_derivative_vec",
    )
    xfails = (
        # not robust
        "horizontal_geodesic_has_horizontal_derivative",
    )
    tolerances = {
        "tangent_vector_projections_orthogonality_with_metric": {"atol": 5e-1},
        "vertical_projection_is_vertical": {"atol": 1e-1},
        "horizontal_projection_is_horizontal": {"atol": 1e-1},
        "horizontal_geodesic_has_horizontal_derivative": {"atol": 1e-1},
        "horizontal_lift_is_horizontal": {"atol": 1e-1},
        "tangent_riemannian_submersion_after_vertical_projection": {"atol": 5e-1},
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
