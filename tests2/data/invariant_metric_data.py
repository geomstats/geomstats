from tests2.data.base_data import RiemannianMetricTestData


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
    skips = (
        # not implemented
        "christoffels_vec",
        "cometric_matrix_vec",
        "covariant_riemann_tensor_bianchi_identity",
        "covariant_riemann_tensor_is_interchange_symmetric",
        "covariant_riemann_tensor_is_skew_symmetric_1",
        "covariant_riemann_tensor_is_skew_symmetric_2",
        "covariant_riemann_tensor_vec",
        "inner_coproduct_vec",
        "inner_product_derivative_matrix_vec",
        "metric_matrix_is_spd",
        "metric_matrix_vec",
        "ricci_tensor_vec",
        "riemann_tensor_vec",
        "scalar_curvature_vec",
        "injectivity_radius_vec",
    )
    tolerances = {
        # basically all the methods that depend on numerical solvers
        "dist_vec": {"atol": 1e-4},
        "dist_is_log_norm": {"atol": 1e-4},
        "dist_is_symmetric": {"atol": 1e-4},
        "dist_triangle_inequality": {"atol": 1e-4},
        "exp_belongs": {"atol": 1e-4},
        "exp_at_identity_vec": {"atol": 1e-4},
        "exp_after_log": {"atol": 1e-4},
        "exp_after_log_at_identity": {"atol": 1e-4},
        "log_after_exp": {"atol": 1e-4},
        "log_vec": {"atol": 1e-4},
        "geodesic_bvp_vec": {"atol": 1e-4},
        "geodesic_boundary_points": {"atol": 1e-4},
        "geodesic_bvp_reverse": {"atol": 1e-4},
        "geodesic_bvp_belongs": {"atol": 1e-4},
        "geodesic_ivp_belongs": {"atol": 1e-4},
        "parallel_transport_transported_is_tangent": {"atol": 1e-4},
        "parallel_transport_vec_with_end_point": {"atol": 1e-4},
        "squared_dist_is_symmetric": {"atol": 1e-4},
        "squared_dist_vec": {"atol": 1e-4},
    }
    xfails = tuple(tolerances.keys())


class InvariantMetricVectorTestData(_InvariantMetricMixinsTestData):
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


class InvariantMetricVectorSOTestData(InvariantMetricVectorTestData):
    skips = (
        # not implemented
        "curvature_derivative_vec",
        "directional_curvature_derivative_vec",
        "parallel_transport_transported_is_tangent",
        "parallel_transport_vec_with_direction",
        "parallel_transport_vec_with_end_point",
        "injectivity_radius_vec",
    )


class BiInvariantMetricTestData(InvariantMetricVectorTestData):
    pass


class BiInvariantMetricSO3VectorTestData(BiInvariantMetricTestData):
    skips = (
        "parallel_transport_transported_is_tangent",
        "parallel_transport_vec_with_end_point",
        "parallel_transport_vec_with_direction",
        # not implemented
        "curvature_derivative_vec",
        "directional_curvature_derivative_vec",
    )


class BiInvariantMetricSOMatrixTestData(BiInvariantMetricTestData):
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
        # not implemented
        "covariant_riemann_tensor_vec",
        "covariant_riemann_tensor_bianchi_identity",
        "covariant_riemann_tensor_is_interchange_symmetric",
        "covariant_riemann_tensor_is_skew_symmetric_1",
        "covariant_riemann_tensor_is_skew_symmetric_2",
        "curvature_derivative_vec",
        "curvature_vec",
        "directional_curvature_derivative_vec",
        "directional_curvature_vec",
        "ricci_tensor_vec",
        "riemann_tensor_vec",
        "scalar_curvature_vec",
        "sectional_curvature_vec",
    )
