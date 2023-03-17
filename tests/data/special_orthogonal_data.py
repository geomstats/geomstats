import itertools
import math
import random
from contextlib import nullcontext as does_not_raise

import pytest

import geomstats.backend as gs
from geomstats.geometry.invariant_metric import BiInvariantMetric, InvariantMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.data_generation import TestData, _InvariantMetricTestData, _LieGroupTestData


def sample_matrix(theta, mul=1.0):
    return gs.array(
        [[gs.cos(theta), mul * gs.sin(theta)], [gs.sin(theta), gs.cos(theta)]]
    )


def sample_algebra_matrix(theta, mul=1.0):
    return mul * gs.array([[0.0, -theta], [theta, 0.0]])


angle_0 = gs.zeros(3)
angle_close_0 = 1e-10 * gs.array([1.0, -1.0, 1.0])
angle_close_pi_low = (gs.pi - 1e-9) / gs.sqrt(2.0) * gs.array([0.0, 1.0, -1.0])
angle_pi = gs.pi * gs.array([1.0, 0, 0])
angle_close_pi_high = (gs.pi + 1e-9) / gs.sqrt(3.0) * gs.array([-1.0, 1.0, -1])
angle_in_pi_2pi = (gs.pi + 0.3) / gs.sqrt(5.0) * gs.array([-2.0, 1.0, 0.0])
angle_close_2pi_low = (2.0 * gs.pi - 1e-9) / gs.sqrt(6.0) * gs.array([2.0, 1.0, -1])
angle_2pi = 2.0 * gs.pi / gs.sqrt(3.0) * gs.array([1.0, 1.0, -1.0])
angle_close_2pi_high = (2.0 * gs.pi + 1e-9) / gs.sqrt(2.0) * gs.array([1.0, 0.0, -1.0])

elements_all = {
    "angle_0": angle_0,
    "angle_close_0": angle_close_0,
    "angle_close_pi_low": angle_close_pi_low,
    "angle_pi": angle_pi,
    "angle_close_pi_high": angle_close_pi_high,
    "angle_in_pi_2pi": angle_in_pi_2pi,
    "angle_close_2pi_low": angle_close_2pi_low,
    "angle_2pi": angle_2pi,
    "angle_close_2pi_high": angle_close_2pi_high,
}


elements = elements_all

coords = ["extrinsic", "intrinsic"]
orders = ["xyz", "zyx"]
angle_pi_6 = gs.pi / 6.0
cos_angle_pi_6 = gs.cos(angle_pi_6)
sin_angle_pi_6 = gs.sin(angle_pi_6)

cos_angle_pi_12 = gs.cos(angle_pi_6 / 2)
sin_angle_pi_12 = gs.sin(angle_pi_6 / 2)

angles_close_to_pi_all = [
    "angle_close_pi_low",
    "angle_pi",
    "angle_close_pi_high",
]

angles_close_to_pi = angles_close_to_pi_all


class SpecialOrthogonalTestData(_LieGroupTestData):
    Space = SpecialOrthogonal

    n_list = random.sample(range(2, 4), 2)
    space_args_list = list(zip(n_list)) + [(2, "vector"), (3, "vector")]
    shape_list = [(n, n) for n in n_list] + [(1,), (3,)]
    n_tangent_vecs_list = random.sample(range(2, 10), 4)
    n_points_list = random.sample(range(2, 10), 4)
    n_vecs_list = random.sample(range(2, 10), 4)

    def belongs_test_data(self):
        theta = gs.pi / 3
        smoke_data = [
            dict(n=2, mat=sample_matrix(theta, mul=-1.0), expected=True),
            dict(n=2, mat=sample_matrix(theta, mul=1.0), expected=False),
            dict(n=2, mat=gs.zeros((2, 3)), expected=False),
            dict(n=3, mat=gs.zeros((2, 3)), expected=False),
            dict(n=2, mat=gs.zeros((2, 2, 3)), expected=gs.array([False, False])),
            dict(
                n=2,
                mat=gs.stack(
                    [
                        sample_matrix(theta / 2, mul=-1.0),
                        sample_matrix(theta / 2, mul=1.0),
                    ]
                ),
                expected=gs.array([True, False]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def dim_test_data(self):
        smoke_data = [
            dict(n=2, expected=1),
            dict(n=3, expected=3),
            dict(n=4, expected=6),
        ]
        return self.generate_tests(smoke_data)

    def identity_test_data(self):
        smoke_data = [
            dict(n=2, point_type="matrix", expected=gs.eye(2)),
            dict(n=3, point_type="matrix", expected=gs.eye(3)),
            dict(n=4, point_type="matrix", expected=gs.eye(4)),
            dict(n=2, point_type="vector", expected=gs.zeros(1)),
        ]
        return self.generate_tests(smoke_data)

    def is_tangent_test_data(self):
        theta = gs.pi / 3
        point = SpecialOrthogonal(2).random_uniform()
        vec_1 = gs.array([[0.0, -theta], [theta, 0.0]])
        vec_2 = gs.array([[0.0, -theta], [theta, 1.0]])
        smoke_data = [
            dict(n=2, vec=vec_1, base_point=None, expected=True),
            dict(n=2, vec=vec_2, base_point=None, expected=False),
            dict(
                n=2,
                vec=gs.stack([vec_1, vec_2]),
                base_point=None,
                expected=[True, False],
            ),
            dict(
                n=2,
                vec=SpecialOrthogonal(2).compose(point, vec_1),
                base_point=point,
                expected=True,
            ),
            dict(
                n=2,
                vec=SpecialOrthogonal(2).compose(point, vec_2),
                base_point=point,
                expected=False,
            ),
        ]
        return self.generate_tests(smoke_data)

    def is_tangent_compose_test_data(self):
        point = SpecialOrthogonal(2).random_uniform()
        theta = 1.0
        vec_1 = gs.array([[0.0, -theta], [theta, 0.0]])
        vec_2 = gs.array([[0.0, -theta], [theta, 1.0]])

        smoke_data = [
            dict(
                n=2,
                vec=SpecialOrthogonal(2).compose(point, vec_1),
                point=point,
                expected=True,
            ),
            dict(
                n=2,
                vec=SpecialOrthogonal(2).compose(point, vec_2),
                point=point,
                expected=False,
            ),
            dict(
                n=2,
                vec=[
                    SpecialOrthogonal(2).compose(point, vec_1),
                    SpecialOrthogonal(2).compose(point, vec_2),
                ],
                point=point,
                expected=[True, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def to_tangent_test_data(self):
        theta = 1.0
        smoke_data = [
            dict(
                n=2,
                vec=[[0.0, -theta], [theta, 0.0]],
                base_point=None,
                expected=[[0.0, -theta], [theta, 0.0]],
            ),
            dict(
                n=2,
                vec=[[1.0, -math.pi], [math.pi, 1.0]],
                base_point=[
                    [gs.cos(math.pi), -1 * gs.sin(math.pi)],
                    [gs.sin(math.pi), gs.cos(math.pi)],
                ],
                expected=[[0.0, -math.pi], [math.pi, 0.0]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def skew_to_vector_and_vector_to_skew_test_data(self):
        random_data = []
        random_data += [
            dict(
                n=2,
                point_type="vector",
                vec=SpecialOrthogonal(2, "vector").random_point(),
            )
        ]
        random_data += [
            dict(
                n=3,
                point_type="vector",
                vec=SpecialOrthogonal(3, "vector").random_point(),
            )
        ]
        return self.generate_tests([], random_data)

    def are_antipodals_test_data(self):
        mat1 = gs.eye(3)
        mat2 = gs.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        smoke_data = [
            dict(n=3, mat1=mat1, mat2=mat2, expected=True),
            dict(
                n=3,
                mat1=gs.array([mat1, mat2]),
                mat2=gs.array([mat2, mat2]),
                expected=[True, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def log_at_antipodals_value_error_test_data(self):
        smoke_data = [
            dict(
                n=3,
                point=gs.eye(3),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
                ),
                expected=pytest.raises(ValueError),
            ),
            dict(
                n=3,
                point=SpecialOrthogonal(3).random_uniform(),
                base_point=SpecialOrthogonal(3).random_uniform(),
                expected=does_not_raise(),
            ),
        ]
        return self.generate_tests(smoke_data)

    def from_vector_from_matrix_test_data(self):
        n_list = [2, 3]
        n_samples_list = random.sample(range(1, 20), 10)
        random_data = [
            dict(n=n, n_samples=n_samples)
            for n, n_samples in zip(n_list, n_samples_list)
        ]
        return self.generate_tests([], random_data)

    def rotation_vector_from_matrix_test_data(self):
        angle = 0.12
        smoke_data = [
            dict(
                n=3,
                point_type="vector",
                point=gs.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, gs.cos(angle), -gs.sin(angle)],
                        [0, gs.sin(angle), gs.cos(angle)],
                    ]
                ),
                expected=0.12 * gs.array([1.0, 0.0, 0.0]),
            ),
            dict(
                n=2,
                point_type="vector",
                point=gs.array(
                    [
                        [gs.cos(angle), -gs.sin(angle)],
                        [gs.sin(angle), gs.cos(angle)],
                    ]
                ),
                expected=gs.array([0.12]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def distance_broadcast_shape_test_data(self):
        n_list = [2, 3]
        n_samples_list = random.sample(range(1, 20), 2)
        smoke_data = [
            dict(n=n, n_samples=n_samples)
            for n, n_samples in zip(n_list, n_samples_list)
        ]
        return self.generate_tests(smoke_data)

    def projection_test_data(self):
        n_list = [2, 3]
        smoke_data = [
            dict(
                n=n,
                point_type="vector",
                mat=gs.eye(n) + 1e-12 * gs.ones((n, n)),
                expected=gs.eye(n),
            )
            for n in n_list
        ]
        return self.generate_tests(smoke_data)

    def projection_shape_test_data(self):
        n_list = [2, 3]
        n_samples_list = random.sample(range(2, 20), 2)
        random_data = [
            dict(
                n=n,
                point_type="matrix",
                n_samples=n_samples,
                expected=(n_samples, n, n),
            )
            for n, n_samples in zip(n_list, n_samples_list)
        ]
        return self.generate_tests([], random_data)

    def skew_matrix_from_vector_test_data(self):
        smoke_data = [dict(n=2, vec=[0.9], expected=[[0.0, -0.9], [0.9, 0.0]])]
        return self.generate_tests(smoke_data)

    def rotation_vector_rotation_matrix_regularize_test_data(self):
        n_list = [2, 3]
        random_data = [
            dict(
                n=n,
                point=SpecialOrthogonal(n=n, point_type="vector").random_point(),
            )
            for n in n_list
        ]
        return self.generate_tests([], random_data)

    def parallel_transport_test_data(self):
        n_list = random.sample(range(2, 10), 5)
        n_samples_list = random.sample(range(2, 10), 5)
        random_data = [
            dict(n=n, n_samples=n_samples)
            for n, n_samples in zip(n_list, n_samples_list)
        ]
        return self.generate_tests([], random_data)

    def matrix_from_rotation_vector_test_data(self):

        rot_vec_3 = 1e-11 * gs.array([12.0, 1.0, -81.0])
        angle = gs.linalg.norm(rot_vec_3)
        skew_rot_vec_3 = 1e-11 * gs.array(
            [[0.0, 81.0, 1.0], [-81.0, 0.0, -12.0], [-1.0, 12.0, 0.0]]
        )
        coef_1 = gs.sin(angle) / angle
        coef_2 = (1.0 - gs.cos(angle)) / (angle**2)
        expected_3 = (
            gs.eye(3)
            + coef_1 * skew_rot_vec_3
            + coef_2 * gs.matmul(skew_rot_vec_3, skew_rot_vec_3)
        )

        rot_vec_4 = gs.array([0.1, 1.3, -0.5])
        angle = gs.linalg.norm(rot_vec_4)
        skew_rot_vec_4 = gs.array(
            [[0.0, 0.5, 1.3], [-0.5, 0.0, -0.1], [-1.3, 0.1, 0.0]]
        )

        coef_1 = gs.sin(angle) / angle
        coef_2 = (1 - gs.cos(angle)) / (angle**2)
        expected_4 = (
            gs.eye(3)
            + coef_1 * skew_rot_vec_4
            + coef_2 * gs.matmul(skew_rot_vec_4, skew_rot_vec_4)
        )
        smoke_data = [
            dict(n=3, rot_vec=gs.array([0.0, 0.0, 0.0]), expected=gs.eye(3)),
            dict(
                n=3,
                rot_vec=gs.array([gs.pi / 3.0, 0.0, 0.0]),
                expected=gs.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 0.5, -gs.sqrt(3.0) / 2],
                        [0.0, gs.sqrt(3.0) / 2, 0.5],
                    ]
                ),
            ),
            dict(n=3, rot_vec=rot_vec_3, expected=expected_3),
            dict(n=3, rot_vec=rot_vec_4, expected=expected_4),
            dict(
                n=2,
                rot_vec=gs.array([gs.pi / 3]),
                expected=gs.array(
                    [[1.0 / 2, -gs.sqrt(3.0) / 2], [gs.sqrt(3.0) / 2, 1.0 / 2]]
                ),
            ),
        ]
        return self.generate_tests(smoke_data)

    def compose_with_inverse_is_identity_test_data(self):
        smoke_data = []
        for space_args in list(zip(self.n_list)):
            smoke_data += [dict(space_args=space_args)]
        return self.generate_tests(smoke_data)

    def compose_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point_type="vector",
                point_a=gs.array([0.12]),
                point_b=gs.array([-0.15]),
                expected=gs.array([-0.03]),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point_type="vector",
                point=gs.array([2 * gs.pi / 5]),
                base_point=gs.array([gs.pi / 5]),
                expected=gs.array([1 * gs.pi / 5]),
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point_type="vector",
                tangent_vec=gs.array([2 * gs.pi / 5]),
                base_point=gs.array([gs.pi / 5]),
                expected=gs.array([3 * gs.pi / 5]),
            )
        ]
        return self.generate_tests(smoke_data)

    def compose_shape_test_data(self):
        smoke_data = [
            dict(n=3, point_type="vector", n_samples=4),
            dict(n=2, point_type="matrix", n_samples=4),
            dict(n=3, point_type="matrix", n_samples=4),
            dict(n=4, point_type="matrix", n_samples=4),
        ]
        return self.generate_tests(smoke_data)

    def rotation_vector_and_rotation_matrix_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point_type="vector",
                rot_vec=gs.array([[2.0], [1.3], [0.8], [0.03]]),
            ),
            dict(n=2, point_type="vector", rot_vec=gs.array([0.78])),
        ]
        return self.generate_tests(smoke_data)

    def regularize_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point_type="vector",
                angle=gs.array([2 * gs.pi + 1]),
                expected=gs.array([1.0]),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=100.0)


class SpecialOrthogonal3TestData(TestData):
    Space = SpecialOrthogonal

    def tait_bryan_angles_matrix_test_data(self):
        xyz = gs.array(
            [
                [
                    [cos_angle_pi_6, -sin_angle_pi_6, 0.0],
                    [sin_angle_pi_6, cos_angle_pi_6, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [cos_angle_pi_6, 0.0, sin_angle_pi_6],
                    [0.0, 1.0, 0.0],
                    [-sin_angle_pi_6, 0.0, cos_angle_pi_6],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [0.0, cos_angle_pi_6, -sin_angle_pi_6],
                    [0.0, sin_angle_pi_6, cos_angle_pi_6],
                ],
            ]
        )
        zyx = gs.flip(xyz, axis=0)
        data = {"xyz": xyz, "zyx": zyx}
        smoke_data = []

        for extrinsic, order in itertools.product([False, True], orders):
            for i in range(3):
                vec = gs.squeeze(gs.array_from_sparse([(0, i)], [angle_pi_6], (1, 3)))
                zyx = order == "zyx"
                smoke_data += [
                    dict(extrinsic=extrinsic, zyx=zyx, vec=vec, mat=data[order][i])
                ]
            smoke_data += [
                dict(extrinsic=extrinsic, zyx=zyx, vec=gs.zeros(3), mat=gs.eye(3))
            ]
        return self.generate_tests(smoke_data)

    def tait_bryan_angles_quaternion_test_data(self):
        xyz = gs.array(
            [
                [cos_angle_pi_12, 0.0, 0.0, sin_angle_pi_12],
                [cos_angle_pi_12, 0.0, sin_angle_pi_12, 0.0],
                [cos_angle_pi_12, sin_angle_pi_12, 0.0, 0.0],
            ]
        )

        zyx = gs.flip(xyz, axis=0)
        data = {"xyz": xyz, "zyx": zyx}
        smoke_data = []
        e1 = gs.array([1.0, 0.0, 0.0, 0.0])
        for extrinsic, order in itertools.product([False, True], orders):
            for i in range(3):
                vec = gs.squeeze(gs.array_from_sparse([(0, i)], [angle_pi_6], (1, 3)))
                zyx = order == "zyx"
                smoke_data += [
                    dict(extrinsic=extrinsic, zyx=zyx, vec=vec, quat=data[order][i])
                ]
            smoke_data += [dict(extrinsic=extrinsic, zyx=zyx, vec=gs.zeros(3), quat=e1)]
        return self.generate_tests(smoke_data)

    def quaternion_from_rotation_vector_tait_bryan_angles_test_data(self):
        smoke_data = []
        for coord, order in itertools.product(coords, orders):
            for angle_type in elements:
                point = elements[angle_type]
                if angle_type not in angles_close_to_pi:
                    smoke_data += [dict(coord=coord, order=order, point=point)]

        return self.generate_tests(smoke_data)

    def tait_bryan_angles_rotation_vector_test_data(self):
        smoke_data = []
        for coord, order in itertools.product(coords, orders):
            for angle_type in elements:
                point = elements[angle_type]
                if angle_type not in angles_close_to_pi:
                    smoke_data += [dict(coord=coord, order=order, point=point)]

        return self.generate_tests(smoke_data)

    def quaternion_and_rotation_vector_with_angles_close_to_pi_test_data(self):
        smoke_data = []
        angle_types = angles_close_to_pi
        for angle_type in angle_types:
            point = elements_all[angle_type]
            smoke_data += [dict(point=point)]

        return self.generate_tests(smoke_data)

    def quaternion_and_matrix_with_angles_close_to_pi_test_data(self):
        smoke_data = []
        angle_types = angles_close_to_pi
        for angle_type in angle_types:
            point = elements_all[angle_type]
            smoke_data += [dict(point=point)]

        return self.generate_tests(smoke_data)

    def rotation_vector_and_rotation_matrix_with_angles_close_to_pi_test_data(self):
        smoke_data = []
        angle_types = angles_close_to_pi
        for angle_type in angle_types:
            point = elements_all[angle_type]
            smoke_data += [dict(point=point)]

        return self.generate_tests(smoke_data)

    def lie_bracket_test_data(self):
        group = SpecialOrthogonal(3, point_type="vector")
        smoke_data = [
            dict(
                tangent_vec_a=gs.array([0.0, 0.0, -1.0]),
                tangent_vec_b=gs.array([0.0, 0.0, -1.0]),
                base_point=group.identity,
                expected=gs.zeros(3),
            ),
            dict(
                tangent_vec_a=gs.array([0.0, 0.0, 1.0]),
                tangent_vec_b=gs.array([0.0, 1.0, 0.0]),
                base_point=group.identity,
                expected=gs.array([-1.0, 0.0, 0.0]),
            ),
            dict(
                tangent_vec_a=gs.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
                tangent_vec_b=gs.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
                base_point=gs.array([group.identity, group.identity]),
                expected=gs.array([gs.zeros(3), gs.array([-1.0, 0.0, 0.0])]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def group_exp_after_log_with_angles_close_to_pi_test_data(self):
        smoke_data = []
        for angle_type in angles_close_to_pi:
            for angle_type_base in elements.values():
                smoke_data += [
                    [elements[angle_type], angle_type_base],
                ]
        return self.generate_tests(smoke_data)

    def group_log_after_exp_with_angles_close_to_pi_test_data(self):
        return self.group_exp_after_log_with_angles_close_to_pi_test_data()

    def left_jacobian_vectorization_test_data(self):
        smoke_data = [dict(n_samples=3)]
        return self.generate_tests(smoke_data)

    def left_jacobian_through_its_determinant_test_data(self):
        smoke_data = []
        for angle_type in elements:
            point = elements[angle_type]
            angle = gs.linalg.norm(SpecialOrthogonal(3, "vector").regularize(point))
            if angle_type in [
                "angle_0",
                "angle_close_0",
                "angle_2pi",
                "angle_close_2pi_high",
            ]:
                expected = 1.0 + angle**2 / 12.0 + angle**4 / 240.0
            else:
                expected = angle**2 / (4 * gs.sin(angle / 2) ** 2)
            smoke_data += [dict(point=point, expected=expected)]

        return self.generate_tests(smoke_data)

    def inverse_test_data(self):
        smoke_data = [dict(n_samples=3)]
        return self.generate_tests(smoke_data)

    def compose_and_inverse_test_data(self):
        smoke_data = []
        for point in elements.values():
            smoke_data += [dict(point=point)]
        return self.generate_tests(smoke_data)

    def compose_regularize_test_data(self):
        smoke_data = []
        for element_type in elements:
            point = elements[element_type]
            if element_type not in angles_close_to_pi:
                smoke_data += [dict(point=point)]
        return self.generate_tests(smoke_data)

    def compose_regularize_angles_close_to_pi_test_data(self):
        smoke_data = []
        for element_type in elements:
            point = elements[element_type]
            if element_type in angles_close_to_pi:
                smoke_data += [dict(point=point)]
        return self.generate_tests(smoke_data)

    def regularize_extreme_cases_test_data(self):
        smoke_data = []
        for angle_type in [
            "angle_close_0",
            "angle_close_pi_low",
            "angle_pi",
            "angle_0",
        ]:
            smoke_data += [
                dict(
                    point=elements_all[angle_type],
                    expected=elements_all[angle_type],
                )
            ]
        point = elements_all["angle_close_pi_high"]
        norm = gs.linalg.norm(point)
        smoke_data += [dict(point=point, expected=point / norm * (norm - 2 * gs.pi))]

        for angle_type in ["angle_in_pi_2pi", "angle_close_2pi_low"]:
            point = elements_all[angle_type]
            angle = gs.linalg.norm(point)
            new_angle = gs.pi - (angle - gs.pi)

            point_initial = point
            expected = -(new_angle / angle) * point_initial
            smoke_data += [dict(point=point, expected=expected)]

        smoke_data += [
            dict(
                point=elements_all["angle_2pi"],
                expected=gs.array([0.0, 0.0, 0.0]),
            )
        ]
        point = elements_all["angle_close_2pi_high"]
        angle = gs.linalg.norm(point)
        new_angle = angle - 2 * gs.pi
        expected = new_angle * point / angle
        smoke_data += [dict(point=point, expected=expected)]
        return self.generate_tests(smoke_data)

    def regularize_test_data(self):
        point = (gs.pi + 1e-6) * gs.array(
            [[1.0, 0.0, 0.0], [2, 0.5, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
        )
        expected_2 = (
            point[1] / gs.linalg.norm(point[1]) * (gs.linalg.norm(point[1]) - 2 * gs.pi)
        )
        expected = gs.array(
            [
                [-(gs.pi - 1e-7), 0.0, 0.0],
                expected_2,
                [0.0, 0.0, 0.0],
                [(gs.pi + 1e-7) / 2.0, 0.0, 0.0],
            ]
        )

        smoke_data = [dict(point=point, expected=expected)]
        return self.generate_tests(smoke_data)


class BiInvariantMetricTestData(_InvariantMetricTestData):
    dim_list = random.sample(range(2, 4), 2)

    metric_args_list = [{} for _ in dim_list]
    shape_list = [(dim, dim) for dim in dim_list]
    group_list = space_list = [SpecialOrthogonal(dim, equip=False) for dim in dim_list]

    n_points_list = random.sample(range(1, 4), 2)
    n_tangent_vecs_list = random.sample(range(1, 4), 2)
    n_points_a_list = random.sample(range(1, 4), 2)
    n_points_b_list = [1]
    batch_size_list = random.sample(range(2, 4), 2)
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = BiInvariantMetric

    def log_after_exp_at_identity_test_data(self):
        return super().log_after_exp_at_identity_test_data(amplitude=100.0)

    def exp_after_log_intrinsic_ball_extrinsic_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                x_intrinsic=gs.array([4.0, 0.2]),
                y_intrinsic=gs.array([3.0, 3]),
            )
        ]
        return self.generate_tests(smoke_data)

    def squared_dist_is_less_than_squared_pi_test_data(self):
        smoke_data = []
        for angle_type_1, angle_type_2 in zip(elements, elements):
            smoke_data += [
                dict(
                    group=SpecialOrthogonal(3, "vector", equip=False),
                    point_1=elements[angle_type_1],
                    point_2=elements[angle_type_2],
                )
            ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        group = SpecialOrthogonal(3, "vector", equip=False)

        theta = gs.pi / 5.0
        rot_vec_base_point = theta / gs.sqrt(3.0) * gs.array([1.0, 1.0, 1.0])
        rot_vec_2 = gs.pi / 4 * gs.array([1.0, 0.0, 0.0])
        phi = (gs.pi / 10) / (gs.tan(gs.array(gs.pi / 10)))
        skew = gs.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
        jacobian = (
            phi * gs.eye(3)
            + (1 - phi) / 3 * gs.ones([3, 3])
            + gs.pi / (10 * gs.sqrt(3.0)) * skew
        )
        inv_jacobian = gs.linalg.inv(jacobian)
        expected = group.compose(
            (gs.pi / 5.0) / gs.sqrt(3.0) * gs.array([1.0, 1.0, 1.0]),
            gs.dot(inv_jacobian, rot_vec_2),
        )
        smoke_data = [
            dict(
                group=group,
                tangent_vec=gs.array([0.0, 0.0, 0.0]),
                base_point=rot_vec_base_point,
                expected=rot_vec_base_point,
            ),
            dict(
                group=group,
                tangent_vec=rot_vec_2,
                base_point=rot_vec_base_point,
                expected=expected,
            ),
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        group = SpecialOrthogonal(3, "vector", equip=False)
        theta = gs.pi / 5.0
        rot_vec_base_point = theta / gs.sqrt(3.0) * gs.array([1.0, 1.0, 1.0])
        # Note: the rotation vector for the reference point
        # needs to be regularized.

        # The Logarithm of a point at itself gives 0.
        expected = gs.array([0.0, 0.0, 0.0])

        # General case: this is the inverse test of test 1 for Riemannian exp
        expected = gs.pi / 4 * gs.array([1.0, 0.0, 0.0])
        phi = (gs.pi / 10) / (gs.tan(gs.array(gs.pi / 10)))
        skew = gs.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
        jacobian = (
            phi * gs.eye(3)
            + (1 - phi) / 3 * gs.ones([3, 3])
            + gs.pi / (10 * gs.sqrt(3.0)) * skew
        )
        inv_jacobian = gs.linalg.inv(jacobian)
        aux = gs.dot(inv_jacobian, expected)
        rot_vec_2 = group.compose(rot_vec_base_point, aux)

        smoke_data = [
            dict(
                group=group,
                point=rot_vec_base_point,
                base_point=rot_vec_base_point,
                expected=gs.array([0.0, 0.0, 0.0]),
            ),
            dict(
                group=group,
                point=rot_vec_2,
                base_point=rot_vec_base_point,
                expected=expected,
            ),
        ]
        return self.generate_tests(smoke_data)

    def distance_broadcast_test_data(self):
        smoke_data = [dict(group=SpecialOrthogonal(n=2, equip=False))]
        return self.generate_tests(smoke_data)


class InvariantMetricTestData(TestData):
    Metric = InvariantMetric

    def squared_dist_is_symmetric_test_data(self):
        smoke_data = []
        group = SpecialOrthogonal(3, "vector", equip=False)
        for angle_type_1, angle_type_2, left in zip(elements, elements, [True, False]):
            smoke_data += [
                dict(
                    group=group,
                    metric_mat_at_identity=9 * gs.eye(group.dim),
                    left=left,
                    point_1=elements[angle_type_1],
                    point_2=elements[angle_type_2],
                )
            ]
        return self.generate_tests(smoke_data)
