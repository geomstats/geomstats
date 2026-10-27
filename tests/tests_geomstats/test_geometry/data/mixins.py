import pytest


class ProjectionMixinsTestData:
    def projection_belongs_test_data(self):
        return self.generate_random_data(
            arg_names="point", samplers={"point": "point_to_project"}
        )

    def projection_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="point",
            samplers={"point": "point_to_project"},
            op_name="projection",
            vectorization_type="basic",
        )


class DistMixinsTestData:
    def squared_dist_is_symmetric_test_data(self):
        return self.generate_random_data(
            arg_names=("point_a", "point_b"),
        )

    def squared_dist_is_positive_test_data(self):
        return self.generate_random_data(
            arg_names=("point_a", "point_b"),
        )

    @pytest.mark.redundant
    def dist_is_symmetric_test_data(self):
        return self.generate_random_data(
            arg_names=("point_a", "point_b"),
        )

    @pytest.mark.redundant
    def dist_is_positive_test_data(self):
        return self.generate_random_data(
            arg_names=("point_a", "point_b"),
        )

    def dist_point_to_itself_is_zero_test_data(self):
        return self.generate_random_data(
            arg_names="point",
        )

    def dist_triangle_inequality_test_data(self):
        return self.generate_random_data(
            arg_names=("point_a", "point_b", "point_c"),
        )

    def squared_dist_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("point_a", "point_b"),
            on_metric=True,
        )

    @pytest.mark.redundant
    def dist_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("point_a", "point_b"),
            on_metric=True,
        )


class GeodesicBVPMixinsTestData:
    def geodesic_boundary_points_test_data(self):
        return self.generate_random_data(
            arg_names=(
                "initial_point",
                "end_point",
            ),
        )

    def geodesic_bvp_reverse_test_data(self):
        return self.generate_timed_random_data(
            arg_names=("initial_point", "end_point"),
        )

    def geodesic_bvp_belongs_test_data(self):
        return self.generate_timed_random_data(
            arg_names=("initial_point", "end_point"),
        )

    def geodesic_bvp_vec_test_data(self):
        return self.generate_timed_vectorization_data(
            arg_names=("initial_point", "end_point"),
            op_name="geodesic",
            on_metric=True,
        )
