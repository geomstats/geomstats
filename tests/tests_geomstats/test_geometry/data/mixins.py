import pytest
from polpo.testing.data import TestDatum
from polpo.testing.lazy import LazyValue

from geomstats.test.random import get_random_times


def _point_to_project(data_generator, n_points):
    return data_generator.point_to_project(n_points)


class ProjectionMixinsTestData:
    def projection_belongs_test_data(self):
        data = []

        for n_points in self.point_counts:
            point = LazyValue(
                _point_to_project,
                self.data_generator,
                n_points,
                label=f"n={n_points}",
            )

            data.append(
                TestDatum(
                    {"point": point},
                    marks=(pytest.mark.random,),
                )
            )

        return data

    def projection_vec_test_data(self):
        point = LazyValue(
            _point_to_project,
            self.data_generator,
            1,
            label="n=1",
        )

        expected = LazyValue(
            lambda point: self.space.projection(point),
            point,
        )

        return self._vectorize_datum(
            {"point": point},
            expected,
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
        data = []
        for n_times in self.time_counts:
            time = get_random_times(n_times)
            data_ = self.generate_random_data(
                arg_names=("initial_point", "end_point"),
                time=time,
            )

            data.extend(data_)

        return data

    def geodesic_bvp_belongs_test_data(self):
        data = []
        for n_times in self.time_counts:
            time = get_random_times(n_times)
            data_ = self.generate_random_data(
                arg_names=("initial_point", "end_point"),
                time=time,
            )

            data.extend(data_)

        return data

    def geodesic_bvp_vec_test_data(self):
        data = []
        for n_times in self.time_counts:
            time = get_random_times(n_times)

            data_ = self.generate_vectorization_data(
                arg_names=("initial_point", "end_point"),
                op_name="geodesic",
                op_evaluator=lambda op, time=time, **kwargs: op(**kwargs)(time),
                time=time,
                on_metric=True,
            )

            data.extend(data_)

        return data
