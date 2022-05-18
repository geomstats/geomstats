import random

import geomstats.backend as gs
from geomstats.information_geometry.gamma import GammaDistributions, GammaMetric
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData


class GammaTestData(_OpenSetTestData):
    space = GammaDistributions
    n_list = random.sample(range(2, 5), 2)
    space_args_list = []
    shape_list = [(2,)]
    n_samples_list = random.sample(range(2, 5), 2)
    n_points_list = random.sample(range(1, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    def belongs_test_data(self):
        smoke_data = [
            dict(vec=[0.1, -1.0], expected=False),
            dict(vec=[0.1, 1.0], expected=True),
            dict(vec=[0.0, 1.0, 0.3], expected=False),
            dict(vec=[-1.0, 0.3], expected=False),
            dict(vec=[0.1, 5], expected=True),
        ]
        return self.generate_tests(smoke_data)

    def random_point_test_data(self):
        random_data = [
            dict(point=self.space().random_point(1), expected=(2,)),
            dict(point=self.space().random_point(5), expected=(5, 2)),
        ]
        return self.generate_tests([], random_data)

    def random_point_belongs_test_data(self):
        smoke_space_args_list = []
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
            self.n_vecs_list,
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
            self.n_vecs_list,
            is_tangent_atol=gs.atol,
        )

    def sample_test_data(self):
        smoke_data = [
            dict(point=gs.array([1.0, 1.0]), n_samples=1, expected=()),
            dict(point=gs.array([[0.1, 0.2], [1, 0.1]]), n_samples=2, expected=(2, 2)),
        ]
        return self.generate_tests(smoke_data)

    def point_to_pdf_test_data(self):
        random_data = [
            dict(
                point=self.space().random_point(2),
                n_samples=10,
            ),
            dict(
                point=self.space().random_point(4),
                n_samples=10,
            ),
            dict(
                point=self.space().random_point(1),
                n_samples=10,
            ),
        ]
        return self.generate_tests([], random_data)


class GammaMetricTestData(_RiemannianMetricTestData):
    space = GammaDistributions
    metric = GammaMetric
    n_list = random.sample(range(2, 5), 2)
    metric_args_list = []
    space_list = [GammaDistributions()]
    space_args_list = []
    n_samples_list = random.sample(range(2, 5), 2)
    shape_list = [(2,)]
    n_points_list = random.sample(range(1, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

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
            self.n_vecs_list,
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
            self.n_vecs_list,
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

    def metric_matrix_shape_test_data(self):
        random_data = [
            dict(point=self.space().random_point(), expected=(2, 2)),
            dict(point=self.space().random_point(3), expected=(3, 2, 2)),
            dict(points=self.space().random_point(2), expected=(2, 2, 2)),
        ]
        return self.generate_tests([], random_data)

    def christoffels_vectorization_test_data(self):
        n_points = 2
        points = self.space().random_point(n_points)
        christoffel_1 = self.metric().christoffels(base_point=points[0])
        christoffel_2 = self.metric().christoffels(base_point=points[1])
        expected = gs.stack((christoffel_1, christoffel_2), axis=0)
        random_data = [dict(point=points, expected=expected)]
        return self.generate_tests([], random_data)

    def christoffels_shape_test_data(self):
        random_data = [
            dict(point=self.space().random_point(1), expected=(2, 2, 2)),
            dict(point=self.space().random_point(3), expected=(3, 2, 2, 2)),
            dict(point=self.space().random_point(2), expected=(2, 2, 2, 2)),
        ]
        return self.generate_tests([], random_data)

    def exp_vectorization_test_data(self):
        point = self.space().random_point()
        n_tangent_vecs = 10
        tangent_vecs = self.space().metric.random_unit_tangent_vec(
            base_point=point, n_vectors=n_tangent_vecs
        )
        random_data = [
            dict(point=point, tangent_vecs=tangent_vecs, exp_solver="geomstats"),
            dict(point=point, tangent_vecs=2 * tangent_vecs, exp_solver="lsoda"),
        ]
        return self.generate_tests([], random_data)

    def exp_control_test_data(self):
        n_times = 10
        base_point = self.space().random_point(n_times)
        tangent_vec = self.space().metric.random_unit_tangent_vec(
            base_point=base_point, n_vectors=1
        )
        random_data = [
            dict(
                base_point=base_point, tangent_vec=tangent_vec, exp_solver="geomstats"
            ),
            dict(
                base_point=base_point, tangent_vec=2 * tangent_vec, exp_solver="lsoda"
            ),
        ]
        return self.generate_tests([], random_data)

    def log_control_test_data(self):
        n_times = 3
        base_point = self.space().random_point(n_times)
        tangent_vec = self.space().metric.random_unit_tangent_vec(
            base_point=base_point, n_vectors=1
        )
        random_data = [
            dict(
                base_point=base_point,
                tangent_vec=tangent_vec,
                exp_solver="geomstats",
                log_method="geodesic_shooting",
            ),
            dict(
                base_point=base_point,
                tangent_vec=2 * tangent_vec,
                exp_solver="lsoda",
                log_method="ode_bvp",
            ),
        ]
        return self.generate_tests([], random_data)

    def exp_after_log_control_test_data(self):
        n_times = 3
        base_point = self.space().random_point(n_times)
        tangent_vec = self.space().metric.random_unit_tangent_vec(
            base_point=base_point, n_vectors=1
        )
        tangent_vec = gs.einsum("i,ij->ij", gs.array([1, 2, 3]), tangent_vec)
        end_point = self.space().metric.exp(
            tangent_vec=tangent_vec, base_point=base_point
        )
        random_data = [
            dict(
                base_point=base_point,
                end_point=end_point,
                exp_solver="geomstats",
                log_method="geodesic_shooting",
                atol=1,
            ),
            dict(
                base_point=base_point,
                end_point=end_point,
                exp_solver="lsoda",
                log_method="ode_bvp",
                atol=1,
            ),
        ]
        return self.generate_tests([], random_data)

    def jacobian_christoffels_test_data(self):
        random_data = [
            dict(point=self.space().random_point(2)),
            dict(point=self.space().random_point(2)),
        ]
        return self.generate_tests([], random_data)

    def geodesic_test_data(self):
        random_data = [
            dict(
                base_point=self.space().random_point(),
                norm=2,
                solver="geomstats",
            ),
            dict(
                base_point=self.space().random_point(),
                norm=1,
                solver="geomstats",
            ),
            dict(
                base_point=self.space().random_point(),
                norm=2,
                solver="vp",
            ),
            dict(
                base_point=self.space().random_point(),
                norm=1,
                solver="vp",
            ),
        ]
        return self.generate_tests([], random_data)
