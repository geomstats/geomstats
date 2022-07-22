import random

import geomstats.backend as gs
from geomstats.information_geometry.beta import BetaDistributions, BetaMetric
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData


class BetaDistributionsTestsData(_OpenSetTestData):
    space = BetaDistributions
    space_args_list = [()]
    shape_list = [(2,)]
    n_samples_list = random.sample(range(2, 5), 2)
    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(2, 5), 2)

    def belongs_test_data(self):
        smoke_data = [
            dict(dim=3, vec=[0.1, 1.0, 0.3], expected=True),
            dict(dim=3, vec=[0.1, 1.0], expected=False),
            dict(dim=3, vec=[0.0, 1.0, 0.3], expected=False),
            dict(dim=2, vec=[-1.0, 0.3], expected=False),
        ]
        return self.generate_tests(smoke_data)

    def random_point_test_data(self):
        random_data = [
            dict(point=self.space(2).random_point(1), expected=(2,)),
            dict(point=self.space(3).random_point(5), expected=(5, 3)),
        ]
        return self.generate_tests([], random_data)

    def random_point_belongs_test_data(self):
        smoke_space_args_list = [(), ()]
        smoke_n_points_list = [1, 2]
        return self._random_point_belongs_test_data(
            smoke_space_args_list,
            smoke_n_points_list,
            self.space_args_list,
            self.n_points_list,
        )

    def projection_belongs_test_data(self):
        return self._projection_belongs_test_data(
            self.space_args_list, self.shape_list, self.n_samples_list
        )

    def to_tangent_is_tangent_test_data(self):
        return self._to_tangent_is_tangent_test_data(
            self.space,
            self.space_args_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        )

    def to_tangent_is_tangent_in_ambient_space_test_data(self):
        return self._to_tangent_is_tangent_in_ambient_space_test_data(
            self.space,
            self.space_args_list,
            self.shape_list,
        )

    def random_tangent_vec_is_tangent_test_data(self):
        return self._random_tangent_vec_is_tangent_test_data(
            self.space,
            self.space_args_list,
            self.n_tangent_vecs_list,
            is_tangent_atol=gs.atol,
        )

    def point_to_pdf_test_data(self):
        smoke_data = [dict(x=gs.linspace(0.0, 1.0, 10))]
        return self.generate_tests(smoke_data)

    def point_to_pdf_vectorization_test_data(self):
        smoke_data = [dict(x=gs.linspace(0.0, 1.0, 10))]
        return self.generate_tests(smoke_data)


class BetaMetricTestData(_RiemannianMetricTestData):
    space = BetaDistributions
    metric = BetaMetric
    metric_args_list = [()]
    shape_list = [(2,)]
    space_list = [BetaDistributions()]
    n_samples_list = random.sample(range(2, 5), 2)
    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(2, 5), 2)

    def exp_shape_test_data(self):
        return self._exp_shape_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_samples_list,
        )

    def log_shape_test_data(self):
        return self._log_shape_test_data(
            self.metric_args_list,
            self.space_list,
        )

    def exp_belongs_test_data(self):
        return self._exp_belongs_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_samples_list,
        )

    def log_is_tangent_test_data(self):
        return self._log_is_tangent_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_samples_list,
        )

    def log_after_exp_test_data(self):
        return self._log_after_exp_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_samples_list,
            rtol=0.1,
            atol=0.0,
        )

    def exp_after_log_test_data(self):
        return self._exp_after_log_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_samples_list,
            self.n_tangent_vecs_list,
            rtol=0.1,
            atol=0.0,
        )

    def squared_dist_is_symmetric_test_data(self):
        return self._squared_dist_is_symmetric_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            self.n_points_list,
            0.1,
            0.1,
        )

    def squared_dist_is_positive_test_data(self):
        return self._squared_dist_is_positive_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            self.n_points_list,
            is_positive_atol=gs.atol,
        )

    def dist_is_symmetric_test_data(self):
        return self._dist_is_symmetric_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            self.n_points_list,
            rtol=0.1,
            atol=gs.atol,
        )

    def dist_is_positive_test_data(self):
        return self._dist_is_positive_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            self.n_points_list,
            is_positive_atol=gs.atol,
        )

    def dist_is_norm_of_log_test_data(self):
        return self._dist_is_norm_of_log_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            self.n_points_list,
            rtol=0.1,
            atol=gs.atol,
        )

    def dist_point_to_itself_is_zero_test_data(self):
        return self._dist_point_to_itself_is_zero_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            rtol=gs.rtol,
            atol=1e-5,
        )

    def inner_product_is_symmetric_test_data(self):
        return self._inner_product_is_symmetric_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            rtol=gs.rtol,
            atol=gs.atol,
        )

    def triangle_inequality_of_dist_test_data(self):
        return self._triangle_inequality_of_dist_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            atol=gs.atol * 10000,
        )

    def riemann_tensor_shape_test_data(self):
        return self._riemann_tensor_shape_test_data(
            self.metric_args_list, self.space_list
        )

    def ricci_tensor_shape_test_data(self):
        return self._ricci_tensor_shape_test_data(
            self.metric_args_list, self.space_list
        )

    def scalar_curvature_shape_test_data(self):
        return self._scalar_curvature_shape_test_data(
            self.metric_args_list, self.space_list
        )

    def covariant_riemann_tensor_is_skew_symmetric_1_test_data(self):
        return self._covariant_riemann_tensor_is_skew_symmetric_1_test_data(
            self.metric_args_list, self.space_list, self.n_points_list
        )

    def covariant_riemann_tensor_is_skew_symmetric_2_test_data(self):
        return self._covariant_riemann_tensor_is_skew_symmetric_2_test_data(
            self.metric_args_list, self.space_list, self.n_points_list
        )

    def covariant_riemann_tensor_bianchi_identity_test_data(self):
        return self._covariant_riemann_tensor_bianchi_identity_test_data(
            self.metric_args_list, self.space_list, self.n_points_list
        )

    def covariant_riemann_tensor_is_interchange_symmetric_test_data(self):
        return self._covariant_riemann_tensor_is_interchange_symmetric_test_data(
            self.metric_args_list, self.space_list, self.n_points_list
        )

    def sectional_curvature_shape_test_data(self):
        return self._sectional_curvature_shape_test_data(
            self.metric_args_list,
            self.n_points_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        )

    def metric_matrix_test_data(self):
        smoke_data = [
            dict(
                point=gs.array([1.0, 1.0]),
                expected=gs.array([[1.0, -0.644934066], [-0.644934066, 1.0]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [dict(n_samples=10)]
        return self.generate_tests(smoke_data)

    def christoffels_shape_test_data(self):
        smoke_data = [dict(n_samples=10)]
        return self.generate_tests(smoke_data)

    def sectional_curvature_test_data(self):
        smoke_data = [dict(n_samples=10, atol=1e-8)]
        return self.generate_tests(smoke_data)
