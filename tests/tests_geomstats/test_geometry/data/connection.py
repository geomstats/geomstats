import pytest
from polpo.testing.geometric import GeometricCaseData

import geomstats.backend as gs
from geomstats.test.random import get_random_times

from .mixins import GeodesicBVPMixinsTestData


class ConnectionTestData(GeodesicBVPMixinsTestData, GeometricCaseData):
    def exp_belongs_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def log_is_tangent_test_data(self):
        return self.generate_random_data(
            arg_names=("point", "base_point"),
        )

    def exp_after_log_test_data(self):
        return self.generate_random_data(
            arg_names=("point", "base_point"),
        )

    def log_after_exp_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def geodesic_ivp_belongs_test_data(self):
        data = []
        for n_times in self.time_counts:
            time = get_random_times(n_times)
            data_ = self.generate_random_data(
                arg_names=("initial_point", "initial_tangent_vec"),
                time=time,
            )

            data.extend(data_)

        return data

    def exp_geodesic_ivp_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def parallel_transport_ivp_transported_is_tangent_test_data(self):
        return self.generate_random_data(
            arg_names=(
                "base_point",
                "tangent_vec",
                "direction",
            ),
        )

    def parallel_transport_bvp_transported_is_tangent_test_data(self):
        return self.generate_random_data(
            arg_names=(
                "base_point",
                "tangent_vec",
                "end_point",
            ),
            dependencies={"tangent_vec": "base_point"},
        )

    def exp_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
            on_metric=True,
        )

    def log_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("point", "base_point"),
            on_metric=True,
        )

    def geodesic_ivp_vec_test_data(self):
        data = []
        for n_times in self.time_counts:
            time = get_random_times(n_times)

            data_ = self.generate_vectorization_data(
                arg_names=("initial_point", "initial_tangent_vec"),
                op_name="geodesic",
                op_evaluator=lambda op, time=time, **kwargs: op(**kwargs)(time),
                time=time,
                on_metric=True,
            )

            data.extend(data_)

        return data

    def parallel_transport_ivp_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=(
                "base_point",
                "tangent_vec",
                "direction",
            ),
            op_name="parallel_transport",
            on_metric=True,
        )

    @pytest.mark.redundant
    def parallel_transport_bvp_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=(
                "base_point",
                "tangent_vec",
                "end_point",
            ),
            dependencies={"tangent_vec": "base_point"},
            op_name="parallel_transport",
            on_metric=True,
        )


class ConnectionFromChristoffelsTestData(ConnectionTestData):
    def christoffels_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="base_point",
            on_metric=True,
        )

    def geodesic_equation_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
            on_metric=True,
            vectorization_type="basic",
            op_evaluator=lambda op, base_point, tangent_vec: op(
                gs.stack([base_point, tangent_vec])
            ),
        )

    def riemann_tensor_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="base_point",
            on_metric=True,
        )

    def curvature_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=(
                "base_point",
                "tangent_vec_a",
                "tangent_vec_b",
                "tangent_vec_c",
            ),
            on_metric=True,
        )

    def ricci_tensor_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="base_point",
            on_metric=True,
        )

    def directional_curvature_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=(
                "base_point",
                "tangent_vec_a",
                "tangent_vec_b",
            ),
            on_metric=True,
        )

    def curvature_derivative_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=(
                "base_point",
                "tangent_vec_a",
                "tangent_vec_b",
                "tangent_vec_c",
                "tangent_vec_d",
            ),
            on_metric=True,
        )

    def directional_curvature_derivative_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=(
                "base_point",
                "tangent_vec_a",
                "tangent_vec_b",
            ),
            on_metric=True,
        )


class ConnectionComparisonTestData(GeometricCaseData):
    def exp_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def log_test_data(self):
        return self.generate_random_data(
            arg_names=("point", "base_point"),
        )

    def geodesic_bvp_test_data(self):
        data = []
        for n_times in self.time_counts:
            time = get_random_times(n_times)
            data_ = self.generate_random_data(
                arg_names=("initial_point", "end_point"),
                time=time,
            )

            data.extend(data_)

        return data

    def geodesic_ivp_test_data(self):
        data = []
        for n_times in self.time_counts:
            time = get_random_times(n_times)
            data_ = self.generate_random_data(
                arg_names=("initial_point", "initial_tangent_vec"),
                time=time,
            )

            data.extend(data_)

        return data

    def parallel_transport_ivp_test_data(self):
        return self.generate_random_data(
            arg_names=(
                "base_point",
                "tangent_vec",
                "direction",
            ),
        )

    def parallel_transport_bvp_test_data(self):
        return self.generate_random_data(
            arg_names=(
                "base_point",
                "tangent_vec",
                "end_point",
            ),
            dependencies={"tangent_vec": "base_point"},
        )


class ConnectionFromChristoffelsComparisonTestData(ConnectionComparisonTestData):
    def christoffels_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def riemann_tensor_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def curvature_test_data(self):
        return self.generate_random_data(
            arg_names=(
                "base_point",
                "tangent_vec_a",
                "tangent_vec_b",
                "tangent_vec_c",
            ),
        )

    def ricci_tensor_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def directional_curvature_test_data(self):
        return self.generate_random_data(
            arg_names=(
                "base_point",
                "tangent_vec_a",
                "tangent_vec_b",
            ),
        )

    def curvature_derivative_test_data(self):
        return self.generate_random_data(
            arg_names=(
                "base_point",
                "tangent_vec_a",
                "tangent_vec_b",
                "tangent_vec_c",
                "tangent_vec_d",
            ),
        )

    def directional_curvature_derivative_test_data(self):
        return self.generate_random_data(
            arg_names=(
                "base_point",
                "tangent_vec_a",
                "tangent_vec_b",
            ),
        )
