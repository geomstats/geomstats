from tests2.data.base_data import RiemannianMetricTestData, VectorSpaceTestData


class EuclideanTestData(VectorSpaceTestData):
    def exp_vec_test_data(self):
        return self.generate_vec_data()

    def exp_random_test_data(self):
        return self.generate_random_data()

    def identity_belongs_test_data(self):
        return self.generate_tests([dict()])


class EuclideanMetricTestData(RiemannianMetricTestData):
    skips = (
        # not implemented
        "curvature_derivative_vec",
        "directional_curvature_derivative_vec",
        "injectivity_radius_vec",
    )
    # TODO: create list
    ignores_if_not_autodiff = (
        "christoffels_vec",
        "covariant_riemann_tensor_vec",
        "curvature_vec",
        "directional_curvature_vec",
        "inner_product_derivative_matrix_vec",
        "ricci_tensor_vec",
        "riemann_tensor_vec",
        "scalar_curvature_vec",
        "sectional_curvature_vec",
    )
