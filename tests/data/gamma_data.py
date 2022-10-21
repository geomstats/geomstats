import random

import geomstats.backend as gs
from geomstats.information_geometry.gamma import GammaDistributions, GammaMetric
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData


class GammaTestData(_OpenSetTestData):
    Space = GammaDistributions
    space_args_list = []
    shape_list = [(2,)]
    n_samples_list = random.sample(range(2, 5), 2)
    n_points_list = random.sample(range(1, 5), 3)
    n_tangent_vecs_list = n_vecs_list = random.sample(range(2, 5), 2)
    batch_shape_list = [
        tuple(random.choices(range(2, 10), k=i)) for i in random.sample(range(1, 5), 3)
    ]

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
            dict(point=self.Space().random_point(1), expected=(2,)),
            dict(point=self.Space().random_point(5), expected=(5, 2)),
        ]
        return self.generate_tests([], random_data)

    def sample_test_data(self):
        smoke_data = [
            dict(
                point=gs.array([1.0, 1.0]), n_samples=random.choice(self.n_samples_list)
            ),
            dict(
                point=gs.array([[0.1, 0.2], [1, 0.1]]),
                n_samples=random.choice(self.n_samples_list),
            ),
        ]
        return self.generate_tests(smoke_data)

    def point_to_pdf_test_data(self):
        random_data = [
            dict(
                point=self.Space().random_point(2),
                n_samples=random.choice(self.n_samples_list),
            ),
            dict(
                point=self.Space().random_point(4),
                n_samples=random.choice(self.n_samples_list),
            ),
            dict(
                point=self.Space().random_point(1),
                n_samples=random.choice(self.n_samples_list),
            ),
        ]
        return self.generate_tests([], random_data)

    def maximum_likelihood_fit_test_data(self):
        smoke_data = [
            dict(
                sample=[1, 2, 3, 4],
                expected=[4.26542805, 2.5],
            ),
            dict(
                sample=[[1, 2, 3, 4, 5], [1, 2, 3, 4]],
                expected=[[3.70164381, 3.0], [4.26542805, 2.5]],
            ),
            dict(
                sample=[gs.array([1, 2, 3, 4, 5]), gs.array([1, 2, 3, 4])],
                expected=[[3.70164381, 3.0], [4.26542805, 2.5]],
            ),
            dict(
                sample=[[1, 2, 3, 4]],
                expected=[4.26542805, 2.5],
            ),
            dict(
                sample=gs.array([1, 2, 3, 4]),
                expected=[4.26542805, 2.5],
            ),
            dict(
                sample=[gs.array([1, 2, 3, 4])],
                expected=[4.26542805, 2.5],
            ),
            dict(
                sample=[[1, 2, 3, 4, 5], gs.array([1, 2, 3, 4])],
                expected=[[3.70164381, 3.0], [4.26542805, 2.5]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def natural_to_standard_test_data(self):
        smoke_data = [
            dict(point=gs.array([1.0, 1.0]), expected=gs.array([1.0, 1.0])),
            dict(point=gs.array([1.0, 2.0]), expected=gs.array([1.0, 0.5])),
        ]
        return self.generate_tests(smoke_data)

    def natural_to_standard_vectorization_test_data(self):
        random_data = [
            dict(point=self.Space().random_point(n_points))
            for n_points in self.n_points_list
        ]
        return self.generate_tests([], random_data)

    def standard_to_natural_test_data(self):
        smoke_data = [
            dict(point=gs.array([1.0, 1.0]), expected=gs.array([1.0, 1.0])),
            dict(point=gs.array([1.0, 2.0]), expected=gs.array([1.0, 0.5])),
        ]
        return self.generate_tests(smoke_data)

    def standard_to_natural_vectorization_test_data(self):
        random_data = [
            dict(point=self.Space().random_point(n_points))
            for n_points in self.n_points_list
        ]
        return self.generate_tests([], random_data)

    def tangent_natural_to_standard_test_data(self):
        smoke_data = [
            dict(
                vec=gs.array([2.0, 1.0]),
                point=gs.array([1.0, 2.0]),
                expected=gs.array([2.0, 0.75]),
            ),
            dict(
                vec=gs.array([1.0, 1.0]),
                point=gs.array([1.0, 1.0]),
                expected=gs.array([1.0, 0]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def tangent_natural_to_standard_vectorization_test_data(self):
        random_data = [
            dict(
                vec=self.Space().random_point(n_points),
                point=self.Space().random_point(),
            )
            for n_points in self.n_points_list
        ] + [
            dict(
                vec=self.Space().random_point(n_points),
                point=self.Space().random_point(n_points),
            )
            for n_points in self.n_points_list
        ]
        return self.generate_tests([], random_data)

    def tangent_standard_to_natural_test_data(self):
        smoke_data = [
            dict(
                vec=gs.array([2.0, 1.0]),
                point=gs.array([1.0, 2.0]),
                expected=gs.array([2.0, 0.75]),
            ),
            dict(
                vec=gs.array([1.0, 1.0]),
                point=gs.array([1.0, 1.0]),
                expected=gs.array([1.0, 0]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def tangent_standard_to_natural_vectorization_test_data(self):
        random_data = [
            dict(
                vec=self.Space().random_point(n_points),
                point=self.Space().random_point(),
            )
            for n_points in self.n_points_list
        ] + [
            dict(
                vec=self.Space().random_point(n_points),
                point=self.Space().random_point(n_points),
            )
            for n_points in self.n_points_list
        ]
        return self.generate_tests([], random_data)


class GammaMetricTestData(_RiemannianMetricTestData):
    Space = GammaDistributions
    Metric = GammaMetric
    connection_args_list = metric_args_list = [()]
    space_list = [GammaDistributions()]
    space_args_list = []
    n_samples_list = random.sample(range(2, 5), 2)
    shape_list = [(2,)]
    n_norms_list = random.sample(range(1, 3), 2)
    n_points_a_list = n_points_b_list = n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = n_vecs_list = random.sample(range(2, 5), 2)

    def metric_matrix_shape_test_data(self):
        random_data = [
            dict(point=self.Space().random_point(), expected=(2, 2)),
            dict(point=self.Space().random_point(3), expected=(3, 2, 2)),
            dict(point=self.Space().random_point(2), expected=(2, 2, 2)),
        ]
        return self.generate_tests([], random_data)

    def christoffels_vectorization_test_data(self):
        n_points = 2
        points = self.Space().random_point(n_points)
        christoffel_1 = self.Metric().christoffels(base_point=points[0])
        christoffel_2 = self.Metric().christoffels(base_point=points[1])
        expected = gs.stack((christoffel_1, christoffel_2), axis=0)
        random_data = [dict(point=points, expected=expected)]
        return self.generate_tests([], random_data)

    def christoffels_shape_test_data(self):
        random_data = [
            dict(point=self.Space().random_point(), expected=(2, 2, 2)),
            dict(point=self.Space().random_point(3), expected=(3, 2, 2, 2)),
            dict(point=self.Space().random_point(2), expected=(2, 2, 2, 2)),
        ]
        return self.generate_tests([], random_data)

    def exp_vectorization_test_data(self):
        point = self.Space().random_point()
        n_tangent_vecs = random.choice(self.n_vecs_list)
        tangent_vecs = self.Space().metric.random_unit_tangent_vec(
            base_point=point, n_vectors=n_tangent_vecs
        )
        random_data = [
            dict(
                point=point,
                tangent_vecs=random.choice(self.n_norms_list) * tangent_vecs,
                exp_solver="geomstats",
            ),
            dict(
                point=point,
                tangent_vecs=random.choice(self.n_norms_list) * tangent_vecs,
                exp_solver="lsoda",
            ),
        ]
        return self.generate_tests([], random_data)

    def exp_control_test_data(self):
        n_points = random.choice(self.n_points_list)
        base_point = self.Space().random_point(n_points)
        tangent_vec = self.Space().metric.random_unit_tangent_vec(
            base_point=base_point, n_vectors=1
        )
        random_data = [
            dict(
                base_point=base_point,
                tangent_vec=random.choice(self.n_norms_list) * tangent_vec,
                exp_solver="geomstats",
            ),
            dict(
                base_point=base_point,
                tangent_vec=random.choice(self.n_norms_list) * tangent_vec,
                exp_solver="lsoda",
            ),
        ]
        return self.generate_tests([], random_data)

    def log_control_test_data(self):
        n_points = random.choice(self.n_points_list)
        base_point = self.Space().random_point(n_points, lower_bound=1.0)
        tangent_vec = self.Space().metric.random_unit_tangent_vec(
            base_point=base_point, n_vectors=1
        )
        random_data = [
            dict(
                base_point=base_point,
                tangent_vec=random.choice(self.n_norms_list) * tangent_vec,
                exp_solver="geomstats",
                log_method="geodesic_shooting",
            ),
            dict(
                base_point=base_point,
                tangent_vec=random.choice(self.n_norms_list) * tangent_vec,
                exp_solver="lsoda",
                log_method="ode_bvp",
            ),
            dict(
                base_point=base_point,
                tangent_vec=random.choice(self.n_norms_list) * tangent_vec,
                exp_solver="geomstats",
                log_method="ode_bvp",
            ),
            dict(
                base_point=base_point,
                tangent_vec=random.choice(self.n_norms_list) * tangent_vec,
                exp_solver="lsoda",
                log_method="ode_bvp",
            ),
        ]
        return self.generate_tests([], random_data)

    def exp_after_log_control_test_data(self):
        n_points = random.choice(self.n_points_list)
        base_point = self.Space().random_point(n_points, lower_bound=1.0)
        tangent_vec = self.Space().metric.random_unit_tangent_vec(
            base_point=base_point, n_vectors=1
        )
        tangent_vec = gs.squeeze(
            gs.einsum(
                "...,...j->...j",
                gs.array(random.choices(self.n_norms_list, k=n_points)),
                tangent_vec,
            )
        )
        end_point = self.Space().metric.exp(
            tangent_vec=tangent_vec, base_point=base_point
        )
        random_data = [
            dict(
                base_point=base_point,
                end_point=end_point,
                exp_solver="geomstats",
                log_method="geodesic_shooting",
                rtol=1,
            ),
            dict(
                base_point=base_point,
                end_point=end_point,
                exp_solver="lsoda",
                log_method="ode_bvp",
                rtol=1,
            ),
            dict(
                base_point=base_point,
                end_point=end_point,
                exp_solver="geomstats",
                log_method="ode_bvp",
                rtol=1,
            ),
            dict(
                base_point=base_point,
                end_point=end_point,
                exp_solver="lsoda",
                log_method="geodesic_shooting",
                rtol=1,
            ),
        ]
        return self.generate_tests([], random_data)

    def log_after_exp_control_test_data(self):
        n_points = random.choice(self.n_points_list)
        base_point = self.Space().random_point(n_points, lower_bound=1.0)
        tangent_vec = self.Space().metric.random_unit_tangent_vec(
            base_point=base_point, n_vectors=1
        )
        tangent_vec = gs.squeeze(
            gs.einsum(
                "...,...j->...j",
                gs.array(random.choices(self.n_norms_list, k=n_points)),
                tangent_vec,
            )
        )
        random_data = [
            dict(
                base_point=base_point,
                tangent_vec=tangent_vec,
                exp_solver="geomstats",
                log_method="geodesic_shooting",
                rtol=1,
            ),
            dict(
                base_point=base_point,
                tangent_vec=tangent_vec,
                exp_solver="lsoda",
                log_method="ode_bvp",
                rtol=1,
            ),
            dict(
                base_point=base_point,
                tangent_vec=tangent_vec,
                exp_solver="geomstats",
                log_method="ode_bvp",
                rtol=1,
            ),
            dict(
                base_point=base_point,
                tangent_vec=tangent_vec,
                exp_solver="lsoda",
                log_method="geodesic_shooting",
                rtol=1,
            ),
        ]
        return self.generate_tests([], random_data)

    def jacobian_christoffels_test_data(self):
        random_data = [
            dict(
                point=self.Space().random_point(random.choice(self.n_points_list) + 1)
            ),
        ]
        return self.generate_tests([], random_data)

    def geodesic_test_data(self):
        random_data = [
            dict(
                base_point=self.Space().random_point(),
                norm=random.choice(self.n_norms_list),
                solver="geomstats",
            ),
            dict(
                base_point=self.Space().random_point(),
                norm=random.choice(self.n_norms_list),
                solver="vp",
            ),
        ]
        return self.generate_tests([], random_data)

    def geodesic_shape_test_data(self):
        random_data = [
            dict(
                point=self.Space().random_point(),
                n_vec=1,
                norm=random.choice(self.n_norms_list),
                time=0.5,
                solver="geomstats",
                expected=(1, 2),
            ),
            dict(
                point=self.Space().random_point(),
                n_vec=4,
                norm=random.choice(self.n_norms_list),
                time=0.5,
                solver="geomstats",
                expected=(4, 1, 2),
            ),
            dict(
                point=self.Space().random_point(),
                n_vec=4,
                norm=random.choice(self.n_norms_list),
                time=gs.linspace(0.0, 1.0, 10),
                solver="geomstats",
                expected=(4, 10, 2),
            ),
            dict(
                point=self.Space().random_point(),
                n_vec=1,
                norm=random.choice(self.n_norms_list),
                time=0.5,
                solver="vp",
                expected=(1, 2),
            ),
            dict(
                point=self.Space().random_point(),
                n_vec=4,
                norm=random.choice(self.n_norms_list),
                time=0.5,
                solver="vp",
                expected=(4, 1, 2),
            ),
            dict(
                point=self.Space().random_point(),
                n_vec=4,
                norm=random.choice(self.n_norms_list),
                time=gs.linspace(0.0, 1.0, 10),
                solver="vp",
                expected=(4, 10, 2),
            ),
        ]
        return self.generate_tests([], random_data)

    def scalar_curvature_test_data(self):
        random_data = [
            dict(point=self.Space().random_point()),
            dict(point=self.Space().random_point(2)),
            dict(point=self.Space().random_point(3)),
        ]
        return self.generate_tests([], random_data)
