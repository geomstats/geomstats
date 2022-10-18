import random

import geomstats.backend as gs
from geomstats.geometry.pre_shape import (
    KendallShapeMetric,
    PreShapeMetric,
    PreShapeSpace,
)
from tests.data_generation import _LevelSetTestData, _RiemannianMetricTestData

smoke_space = PreShapeSpace(4, 3)
vector = gs.random.rand(11, 4, 3)
base_point = smoke_space.random_point()
tg_vec_0 = smoke_space.to_tangent(vector[0], base_point)
hor_x = smoke_space.horizontal_projection(tg_vec_0, base_point)
tg_vec_1 = smoke_space.to_tangent(vector[1], base_point)
hor_y = smoke_space.horizontal_projection(tg_vec_1, base_point)
tg_vec_2 = smoke_space.to_tangent(vector[2], base_point)
hor_z = smoke_space.horizontal_projection(tg_vec_2, base_point)
tg_vec_3 = smoke_space.to_tangent(vector[3], base_point)
hor_h = smoke_space.horizontal_projection(tg_vec_3, base_point)
tg_vec_4 = smoke_space.to_tangent(vector[4], base_point)
ver_v = smoke_space.vertical_projection(tg_vec_4, base_point)
tg_vec_5 = smoke_space.to_tangent(vector[5], base_point)
ver_w = smoke_space.vertical_projection(tg_vec_5, base_point)
tg_vec_6 = smoke_space.to_tangent(vector[6], base_point)
hor_dy = smoke_space.horizontal_projection(tg_vec_6, base_point)
tg_vec_7 = smoke_space.to_tangent(vector[7], base_point)
hor_dz = smoke_space.horizontal_projection(tg_vec_7, base_point)
tg_vec_8 = smoke_space.to_tangent(vector[8], base_point)
ver_dv = smoke_space.vertical_projection(tg_vec_8, base_point)
tg_vec_9 = smoke_space.to_tangent(vector[9], base_point)
ver_dw = smoke_space.vertical_projection(tg_vec_9, base_point)
tg_vec_10 = smoke_space.to_tangent(vector[10], base_point)
hor_dh = smoke_space.horizontal_projection(tg_vec_10, base_point)

# generate valid derivatives of horizontal / vertical vector fields.
a_x_y = smoke_space.integrability_tensor(hor_x, hor_y, base_point)
nabla_x_y = hor_dy + a_x_y
a_x_z = smoke_space.integrability_tensor(hor_x, hor_z, base_point)
nabla_x_z = hor_dz + a_x_z
a_x_v = smoke_space.integrability_tensor(hor_x, ver_v, base_point)
nabla_x_v = ver_dv + a_x_v
a_x_w = smoke_space.integrability_tensor(hor_x, ver_w, base_point)
nabla_x_w = ver_dw + a_x_w
a_x_h = smoke_space.integrability_tensor(hor_x, hor_h, base_point)
nabla_x_h = hor_dh + a_x_h


class PreShapeSpaceTestData(_LevelSetTestData):
    k_landmarks_list = random.sample(range(3, 6), 2)
    m_ambient_list = [random.sample(range(2, n), 1)[0] for n in k_landmarks_list]
    space_args_list = list(zip(k_landmarks_list, m_ambient_list))
    n_points_list = random.sample(range(1, 5), 2)
    shape_list = space_args_list
    n_vecs_list = random.sample(range(1, 5), 2)

    Space = PreShapeSpace

    def belongs_test_data(self):
        random_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                mat=gs.random.rand(2, 4),
                expected=gs.array(False),
            ),
            dict(
                m_ambient=3,
                k_landmarks=4,
                mat=gs.random.rand(10, 2, 4),
                expected=gs.array([False] * 10),
            ),
        ]
        return self.generate_tests([], random_data)

    def is_centered_test_data(self):
        random_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                point=gs.ones((4, 3)),
                expected=gs.array(False),
            ),
            dict(
                k_landmarks=4,
                m_ambient=3,
                point=gs.zeros((4, 3)),
                expected=gs.array(True),
            ),
        ]
        return self.generate_tests([], random_data)

    def to_center_is_center_test_data(self):
        smoke_data = [
            dict(k_landmarks=4, m_ambient=3, point=gs.ones((4, 3))),
            dict(k_landmarks=4, m_ambient=3, point=gs.ones((10, 4, 3))),
        ]
        return self.generate_tests(smoke_data)

    def vertical_projection_test_data(self):
        vector = gs.random.rand(10, 4, 3)
        space = self.Space(4, 3)
        point = space.random_point()
        smoke_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                tangent_vec=space.to_tangent(vector[0], point),
                point=point,
            ),
            dict(
                k_landmarks=4,
                m_ambient=3,
                tangent_vec=space.to_tangent(vector, point),
                point=point,
            ),
        ]
        return self.generate_tests(smoke_data)

    def horizontal_projection_test_data(self):
        vector = gs.random.rand(10, 4, 3)
        space = self.Space(4, 3)
        point = space.random_point()
        smoke_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                tangent_vec=space.to_tangent(vector[0], point),
                point=point,
            ),
            dict(
                k_landmarks=4,
                m_ambient=3,
                tangent_vec=space.to_tangent(vector, point),
                point=point,
            ),
        ]
        return self.generate_tests(smoke_data)

    def horizontal_and_is_tangent_test_data(self):
        vector = gs.random.rand(10, 4, 3)
        space = self.Space(4, 3)
        point = space.random_point()
        smoke_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                tangent_vec=space.to_tangent(vector[0], point),
                point=point,
            ),
            dict(
                k_landmarks=4,
                m_ambient=3,
                tangent_vec=space.to_tangent(vector, point),
                point=point,
            ),
        ]
        return self.generate_tests(smoke_data)

    def alignment_is_symmetric_test_data(self):
        space = self.Space(4, 3)
        random_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                point=space.random_point(),
                base_point=space.random_point(),
            ),
            dict(
                k_landmarks=4,
                m_ambient=3,
                point=space.random_point(),
                base_point=space.random_point(2),
            ),
            dict(
                k_landmarks=4,
                m_ambient=3,
                point=space.random_point(2),
                base_point=space.random_point(2),
            ),
        ]
        return self.generate_tests([], random_data)

    def integrability_tensor_test_data(self):
        space = self.Space(4, 3)
        vector = gs.random.rand(2, 4, 3)
        base_point = space.random_point()
        random_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                tangent_vec_a=space.to_tangent(vector[0], base_point),
                tangent_vec_b=space.to_tangent(vector[1], base_point),
                base_point=base_point,
            )
        ]
        return self.generate_tests(random_data)

    def integrability_tensor_old_test_data(self):
        return self.integrability_tensor_test_data()

    def integrability_tensor_derivative_is_alternate_test_data(self):
        smoke_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                hor_x=hor_x,
                hor_y=hor_y,
                hor_z=hor_z,
                nabla_x_y=nabla_x_y,
                nabla_x_z=nabla_x_z,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def integrability_tensor_derivative_is_skew_symmetric_test_data(self):
        smoke_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                hor_x=hor_x,
                hor_y=hor_y,
                hor_z=hor_z,
                ver_v=ver_v,
                nabla_x_y=nabla_x_y,
                nabla_x_z=nabla_x_z,
                nabla_x_v=nabla_x_v,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def integrability_tensor_derivative_reverses_hor_ver_test_data(self):
        smoke_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                hor_x=hor_x,
                hor_y=hor_y,
                hor_z=hor_z,
                ver_v=ver_v,
                ver_w=ver_w,
                hor_h=hor_h,
                nabla_x_y=nabla_x_y,
                nabla_x_z=nabla_x_z,
                nabla_x_h=nabla_x_h,
                nabla_x_v=nabla_x_v,
                nabla_x_w=nabla_x_w,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def integrability_tensor_derivative_parallel_test_data(self):
        smoke_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                hor_x=hor_x,
                hor_y=hor_y,
                hor_z=hor_z,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def iterated_integrability_tensor_derivative_parallel_test_data(self):
        smoke_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                hor_x=hor_x,
                hor_y=hor_y,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)


class KendallShapeMetricTestData(_RiemannianMetricTestData):
    k_landmarks_list = random.sample(range(3, 6), 2)
    m_ambient_list = [random.sample(range(2, n), 1)[0] for n in k_landmarks_list]
    metric_args_list = list(zip(k_landmarks_list, m_ambient_list))

    shape_list = metric_args_list
    space_list = [PreShapeSpace(k, m) for k, m in metric_args_list]
    n_points_list = random.sample(range(1, 4), 2)
    n_samples_list = random.sample(range(1, 4), 2)
    n_points_a_list = random.sample(range(1, 4), 2)
    n_points_b_list = [1]
    n_tangent_vecs_list = random.sample(range(1, 4), 2)
    batch_size_list = random.sample(range(2, 4), 2)
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = KendallShapeMetric
    Space = PreShapeSpace

    def curvature_is_skew_operator_test_data(self):
        base_point = smoke_space.random_point(2)
        vec = gs.random.rand(4, 4, 3)
        smoke_data = [dict(k_landmarks=4, m_ambient=3, vec=vec, base_point=base_point)]
        return self.generate_tests(smoke_data)

    def curvature_bianchi_identity_test_data(self):
        smoke_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                tangent_vec_a=tg_vec_0,
                tangent_vec_b=tg_vec_1,
                tangent_vec_c=tg_vec_2,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def kendall_sectional_curvature_test_data(self):
        k_landmarks = 4
        m_ambient = 3
        space = smoke_space
        n_samples = 4 * k_landmarks * m_ambient
        base_point = space.random_point(1)

        vec_a = gs.random.rand(n_samples, k_landmarks, m_ambient)
        tg_vec_a = space.to_tangent(space.center(vec_a), base_point)

        vec_b = gs.random.rand(n_samples, k_landmarks, m_ambient)
        tg_vec_b = space.to_tangent(space.center(vec_b), base_point)

        smoke_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                tangent_vec_a=tg_vec_a,
                tangent_vec_b=tg_vec_b,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def kendall_curvature_derivative_bianchi_identity_test_data(self):
        smoke_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                hor_x=hor_x,
                hor_y=hor_y,
                hor_z=hor_z,
                hor_h=hor_h,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def curvature_derivative_is_skew_operator_test_data(self):
        smoke_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                hor_x=hor_x,
                hor_y=hor_y,
                hor_z=hor_z,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def directional_curvature_derivative_test_data(self):
        smoke_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                hor_x=hor_x,
                hor_y=hor_y,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def directional_curvature_derivative_is_quadratic_test_data(self):
        coef_x = -2.5
        coef_y = 1.5
        smoke_data = [
            dict(
                k_landmarks=4,
                m_ambient=3,
                coef_x=coef_x,
                coef_y=coef_y,
                hor_x=hor_x,
                hor_y=hor_y,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def parallel_transport_test_data(self):
        k_landmarks = 4
        m_ambient = 3
        n_samples = 10
        space = PreShapeSpace(4, 3)
        base_point = space.projection(gs.eye(4)[:, :3])
        vec_a = gs.random.rand(n_samples, k_landmarks, m_ambient)
        tangent_vec_a = space.to_tangent(space.center(vec_a), base_point)

        vec_b = gs.random.rand(n_samples, k_landmarks, m_ambient)
        tangent_vec_b = space.to_tangent(space.center(vec_b), base_point)
        smoke_data = [
            dict(
                k_landmarks=k_landmarks,
                m_ambient=m_ambient,
                tangent_vec_a=tangent_vec_a,
                tangent_vec_b=tangent_vec_b,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)


class PreShapeMetricTestData(_RiemannianMetricTestData):
    k_landmarks_list = random.sample(range(3, 6), 2)
    m_ambient_list = [random.sample(range(2, n), 1)[0] for n in k_landmarks_list]
    metric_args_list = list(zip(k_landmarks_list, m_ambient_list))

    shape_list = metric_args_list
    space_list = [PreShapeSpace(k, m) for k, m in metric_args_list]
    n_points_list = random.sample(range(1, 7), 2)
    n_samples_list = random.sample(range(1, 7), 2)
    n_points_a_list = random.sample(range(1, 7), 2)
    n_points_b_list = [1]
    batch_size_list = random.sample(range(2, 7), 2)
    n_tangent_vecs_list = random.sample(range(2, 7), 2)
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = Connection = PreShapeMetric
