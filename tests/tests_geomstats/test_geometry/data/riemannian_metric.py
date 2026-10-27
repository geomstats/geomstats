import pytest

from .connection import (
    ConnectionComparisonTestData,
    ConnectionFromChristoffelsComparisonTestData,
    ConnectionFromChristoffelsTestData,
    ConnectionTestData,
)
from .mixins import DistMixinsTestData


class RiemannianMetricTestData(DistMixinsTestData, ConnectionTestData):
    def inner_product_is_symmetric_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b"),
        )

    def norm_is_positive_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "vector"),
        )

    def normalize_is_unitary_test_data(self):
        return self.generate_random_data(
            arg_names=("point", "vec"),
        )

    def dist_is_log_norm_test_data(self):
        return self.generate_random_data(
            arg_names=("point_a", "point_b"),
        )

    def parallel_transport_ivp_norm_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec", "direction")
        )

    def parallel_transport_bvp_norm_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec", "end_point"),
            dependencies={"tangent_vec": "base_point"},
        )

    def inner_product_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b"),
            on_metric=True,
        )

    @pytest.mark.redundant
    def squared_norm_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "vector"),
            on_metric=True,
        )

    @pytest.mark.redundant
    def norm_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "vector"),
            on_metric=True,
        )

    def normalize_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "vector"),
            on_metric=True,
        )

    def injectivity_radius_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="base_point",
            on_metric=True,
        )


class RiemannianMetricFromMatrixTestData(
    ConnectionFromChristoffelsTestData, RiemannianMetricTestData
):
    def metric_matrix_is_spd_test_data(self):
        return self.generate_random_data(
            arg_names="base_point",
        )

    def covariant_riemann_tensor_is_skew_symmetric_1_test_data(self):
        return self.generate_random_data(
            arg_names="base_point",
        )

    def covariant_riemann_tensor_is_skew_symmetric_2_test_data(self):
        return self.generate_random_data(
            arg_names="base_point",
        )

    def covariant_riemann_tensor_bianchi_identity_test_data(self):
        return self.generate_random_data(
            arg_names="base_point",
        )

    def covariant_riemann_tensor_is_interchange_symmetric_test_data(self):
        return self.generate_random_data(
            arg_names="base_point",
        )

    def metric_matrix_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="base_point",
            on_metric=True,
        )

    def inner_product_derivative_matrix_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="base_point",
            on_metric=True,
        )

    def cometric_matrix_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="base_point",
            on_metric=True,
        )

    def covariant_riemann_tensor_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="base_point",
            on_metric=True,
        )

    def sectional_curvature_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b"),
            on_metric=True,
        )

    def scalar_curvature_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="base_point",
            on_metric=True,
        )


class RiemannianMetricComparisonTestData(ConnectionComparisonTestData):
    # TODO: skipping inner_coproduct

    def inner_product_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )

    @pytest.mark.redundant
    def squared_norm_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "vector"))

    @pytest.mark.redundant
    def norm_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "vector"))

    def normalize_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "vector"))

    @pytest.mark.redundant
    def squared_dist_test_data(self):
        return self.generate_random_data(arg_names=("point_a", "point_b"))

    @pytest.mark.redundant
    def dist_test_data(self):
        return self.generate_random_data(arg_names=("point_a", "point_b"))

    def injectivity_radius_test_data(self):
        return self.generate_random_data(arg_names="base_point")


class RiemannianMetricFromMatrixComparisonTestData(
    RiemannianMetricComparisonTestData,
    ConnectionFromChristoffelsComparisonTestData,
):
    def metric_matrix_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def inner_product_derivative_matrix_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def cometric_matrix_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def covariant_riemann_tensor_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def sectional_curvature_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )

    def scalar_curvature_test_data(self):
        return self.generate_random_data(arg_names="base_point")
