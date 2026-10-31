from polpo.testing.geometric import GeometricMapCaseData

from .base import LevelSetTestData
from .fiber_bundle import FiberBundleTestData
from .quotient_metric import QuotientMetricTestData


class FullRankCorrelationMatricesTestData(LevelSetTestData):
    def tangent_submersion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("point", "vector"),
            vectorization_type="sym",
        )

    def from_covariance_belongs_test_data(self):
        return self.generate_random_data(
            arg_names="point", samplers={"point": "point_to_project"}
        )

    def from_covariance_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="point", samplers={"point": "point_to_project"}
        )

    def diag_action_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("diagonal_vec", "point"),
            samplers={
                "diagonal_vec": "random_diagonal_vec",
                "point": "point_to_project",
            },
            vectorization_type="sym",
        )


class CorrelationMatricesBundleTestData(FiberBundleTestData):
    def log_after_align_is_horizontal_test_data(self):
        return self.with_values(
            super().log_after_align_is_horizontal_test_data(), atol=1e-2
        )

    def align_vec_test_data(self):
        return self.with_values(super().align_vec_test_data(), atol=1e-2)

    def horizontal_projection_is_horizontal_v2_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))


class FullRankCorrelationAffineQuotientMetricTestData(QuotientMetricTestData):
    def dist_point_to_itself_is_zero_test_data(self):
        return self.with_values(
            super().dist_point_to_itself_is_zero_test_data(), atol=1e-4
        )

    def geodesic_bvp_vec_test_data(self):
        return self.with_values(super().geodesic_bvp_vec_test_data(), atol=1e-4)

    def geodesic_bvp_reverse_test_data(self):
        return self.with_values(super().geodesic_bvp_reverse_test_data(), atol=1e-4)

    def geodesic_boundary_points_test_data(self):
        return self.with_values(super().geodesic_boundary_points_test_data(), atol=1e-4)

    def log_after_exp_test_data(self):
        return self.with_values(super().log_after_exp_test_data(), atol=1e-3)

    def exp_after_log_test_data(self):
        return self.with_values(super().exp_after_log_test_data(), atol=1e-4)

    def log_vec_test_data(self):
        return self.with_values(super().log_vec_test_data(), atol=1e-4)


class UniqueDiagonalMatrixAlgorithmTestData(GeometricMapCaseData):
    def map_belongs_to_image_test_data(self):
        return self.generate_random_data(
            arg_names="point",
        )

    def map_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="point",
            op_name="__call__",
        )


class SPDScalingFinderTestData(GeometricMapCaseData):
    def rows_sum_to_one_test_data(self):
        return self.generate_random_data(
            arg_names="point",
        )

    def values_are_positive_test_data(self):
        return self.generate_random_data(
            arg_names="point",
        )

    def map_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="point",
            op_name="__call__",
        )
