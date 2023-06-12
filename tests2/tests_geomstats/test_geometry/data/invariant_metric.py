from .riemannian_metric import RiemannianMetricTestData


class _InvariantMetricMixinsTestData(RiemannianMetricTestData):
    def inner_product_at_identity_vec_test_data(self):
        return self.generate_vec_data()


class InvariantMetricMatrixTestData(_InvariantMetricMixinsTestData):
    def structure_constant_vec_test_data(self):
        return self.generate_vec_data()

    def dual_adjoint_vec_test_data(self):
        return self.generate_vec_data()

    def dual_adjoint_structure_constant_test_data(self):
        return self.generate_random_data()

    def connection_at_identity_vec_test_data(self):
        return self.generate_vec_data()

    def connection_vec_test_data(self):
        return self.generate_vec_data()

    def curvature_at_identity_vec_test_data(self):
        return self.generate_vec_data()

    def sectional_curvature_at_identity_vec_test_data(self):
        return self.generate_vec_data()

    def curvature_derivative_at_identity_vec_test_data(self):
        return self.generate_vec_data()

    def exp_at_identity_vec_test_data(self):
        return self.generate_vec_data()

    def log_after_exp_at_identity_test_data(self):
        return self.generate_random_data()

    def exp_after_log_at_identity_test_data(self):
        return self.generate_random_data()


class InvariantMetricMatrixSOTestData(InvariantMetricMatrixTestData):
    trials = 3
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
    tolerances = {
        "dist_triangle_inequality": {"atol": 1e-4},
        "exp_after_log": {"atol": 1e-4},
        "exp_after_log_at_identity": {"atol": 1e-4},
        "geodesic_boundary_points": {"atol": 1e-4},
        "geodesic_bvp_reverse": {"atol": 1e-4},
        "parallel_transport_bvp_transported_is_tangent": {"atol": 1e-4},
        "parallel_transport_bvp_norm": {"atol": 1e-4},
        "parallel_transport_bvp_vec": {"atol": 1e-4},
    }


class InvariantMetricMatrixSETestData(InvariantMetricMatrixTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
    tolerances = {
        # "dist_vec": {"atol": 1e-4},
        # "dist_is_log_norm": {"atol": 1e-4},
        # "dist_point_to_itself_is_zero": {"atol": 1e-4},
        # "dist_is_symmetric": {"atol": 1e-4},
        # "dist_triangle_inequality": {"atol": 1e-4},
        # "exp_belongs": {"atol": 1e-4},
        # "exp_at_identity_vec": {"atol": 1e-4},
        # "exp_after_log": {"atol": 1e-4},
        # "exp_after_log_at_identity": {"atol": 1e-4},
        # "log_after_exp": {"atol": 1e-4},
        # "log_vec": {"atol": 1e-4},
        # "geodesic_bvp_vec": {"atol": 1e-4},
        # "geodesic_boundary_points": {"atol": 1e-4},
        # "geodesic_bvp_reverse": {"atol": 1e-4},
        # "geodesic_bvp_belongs": {"atol": 1e-4},
        # "parallel_transport_bvp_transported_is_tangent": {"atol": 1e-4},
        # "parallel_transport_bvp_norm": {"atol": 1e-4},
        # "parallel_transport_bvp_vec": {"atol": 1e-4},
        # "squared_dist_is_symmetric": {"atol": 1e-4},
        # "squared_dist_vec": {"atol": 1e-4},
    }


class InvariantMetricVectorTestData(_InvariantMetricMixinsTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    def left_exp_from_identity_vec_test_data(self):
        return self.generate_vec_data()

    def left_log_from_identity_vec_test_data(self):
        return self.generate_vec_data()

    def left_exp_from_identity_after_left_log_from_identity_test_data(self):
        return self.generate_random_data()

    def left_log_from_identity_after_left_exp_from_identity_test_data(self):
        return self.generate_random_data()

    def exp_from_identity_vec_test_data(self):
        return self.generate_vec_data()

    def log_from_identity_vec_test_data(self):
        return self.generate_vec_data()

    def exp_from_identity_after_log_from_identity_test_data(self):
        return self.generate_random_data()

    def log_from_identity_after_exp_from_identity_test_data(self):
        return self.generate_random_data()


class BiInvariantMetricTestData(InvariantMetricVectorTestData):
    pass


class BiInvariantMetricVectorSO3TestData(BiInvariantMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
    skips = (
        "parallel_transport_bvp_norm",
        "parallel_transport_ivp_norm",
        "parallel_transport_bvp_transported_is_tangent",
        "parallel_transport_ivp_transported_is_tangent",
        "parallel_transport_bvp_vec",
        "parallel_transport_ivp_vec",
    )


class BiInvariantMetricMatrixSOTestData(BiInvariantMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
    skips = (
        # due to bad inheritance (misses jacobian_translation)
        "christoffels_vec",
        "cometric_matrix_vec",
        "inner_coproduct_vec",
        "inner_product_derivative_matrix_vec",
        "metric_matrix_is_spd",
        "metric_matrix_vec",
        # due to bad inheritance (misses regularize)
        "exp_from_identity_after_log_from_identity",
        "exp_from_identity_vec",
        "left_exp_from_identity_after_left_log_from_identity",
        "left_exp_from_identity_vec",
        "left_log_from_identity_after_left_exp_from_identity",
        "left_log_from_identity_vec",
        "log_from_identity_after_exp_from_identity",
        "log_from_identity_vec",
    )
