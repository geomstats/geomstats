import itertools
import random
from contextlib import nullcontext as does_not_raise

import pytest

import geomstats.backend as gs
import tests.helper as helper
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.conftest import Parametrizer, TestCase, TestData, tf_backend


def sample_matrix(theta, mul=1.0):
    return mul * gs.array(
        [[gs.cos(theta), -gs.sin(theta)], [gs.sin(theta), gs.cos(theta)]]
    )


def sample_algebra_matrix(theta, mul=1.0):
    return mul * gs.array([[0.0, -theta], [theta, 0.0]])


with_angle_0 = gs.zeros(3)
with_angle_close_0 = 1e-10 * gs.array([1.0, -1.0, 1.0])
with_angle_close_pi_low = (gs.pi - 1e-9) / gs.sqrt(2.0) * gs.array([0.0, 1.0, -1.0])
with_angle_pi = gs.pi * gs.array([1.0, 0, 0])
with_angle_close_pi_high = (gs.pi + 1e-9) / gs.sqrt(3.0) * gs.array([-1.0, 1.0, -1])
with_angle_in_pi_2pi = (gs.pi + 0.3) / gs.sqrt(5.0) * gs.array([-2.0, 1.0, 0.0])
with_angle_close_2pi_low = (
    (2.0 * gs.pi - 1e-9) / gs.sqrt(6.0) * gs.array([2.0, 1.0, -1])
)
with_angle_2pi = 2.0 * gs.pi / gs.sqrt(3.0) * gs.array([1.0, 1.0, -1.0])
with_angle_close_2pi_high = (
    (2.0 * gs.pi + 1e-9) / gs.sqrt(2.0) * gs.array([1.0, 0.0, -1.0])
)

elements_all = {
    "with_angle_0": with_angle_0,
    "with_angle_close_0": with_angle_close_0,
    "with_angle_close_pi_low": with_angle_close_pi_low,
    "with_angle_pi": with_angle_pi,
    "with_angle_close_pi_high": with_angle_close_pi_high,
    "with_angle_in_pi_2pi": with_angle_in_pi_2pi,
    "with_angle_close_2pi_low": with_angle_close_2pi_low,
    "with_angle_2pi": with_angle_2pi,
    "with_angle_close_2pi_high": with_angle_close_2pi_high,
}


coords = ["extrinsic", "intrinsic"]
orders = ["xyz", "zyx"]
angle_pi_6 = gs.pi / 6.0
cos_angle_pi_6 = gs.cos(angle_pi_6)
sin_angle_pi_6 = gs.sin(angle_pi_6)

cos_angle_pi_12 = gs.cos(angle_pi_6 / 2)
sin_angle_pi_12 = gs.sin(angle_pi_6 / 2)

angles_close_to_pi_all = [
    "with_angle_close_pi_low",
    "with_angle_pi",
    "with_angle_close_pi_high",
]

angles_close_to_pi = angles_close_to_pi_all

if tf_backend():
    angles_close_to_pi = ["with_angle_close_pi_low"]


class TestSpecialOrthogonal(TestCase, metaclass=Parametrizer):
    cls = SpecialOrthogonal

    class TestDataSpecialOrthogonal(TestData):
        def belongs_data(self):
            theta = gs.pi / 3
            smoke_data = [
                dict(n=2, mat=sample_matrix(theta), expected=True),
                dict(n=2, mat=sample_matrix(theta, mul=2.0), expected=False),
                dict(n=2, mat=gs.zeros((2, 3)), expected=False),
                dict(n=3, mat=gs.zeros((2, 3)), expected=False),
                dict(n=2, mat=gs.zeros((2, 2, 3)), expected=False),
                dict(
                    n=2,
                    mat=[sample_matrix(theta / 2), sample_matrix(theta / 2, 2)],
                    expected=[True, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def dim_data(self):
            smoke_data = [
                dict(n=2, expected=1),
                dict(n=3, expected=3),
                dict(n=4, expected=6),
            ]
            return self.generate_tests(smoke_data)

        def identity_data(self):
            smoke_data = [
                dict(n=2, expected=gs.eye(2)),
                dict(n=3, expected=gs.eye(3)),
                dict(n=4, expected=gs.eye(4)),
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_data(self):
            n_list = random.sample(range(2, 50), 10)
            smoke_data = [
                dict(n=2, n_samples=100),
                dict(n=3, n_samples=100),
                dict(n=10, n_samples=100),
            ]
            random_data = [dict(n=n, n_samples=100) for n in n_list]
            return self.generate_tests(smoke_data, random_data)

        def is_tangent_data(self):
            theta = gs.pi / 3
            smoke_data = [
                dict(n=2, vec=[[0.0, -theta], [theta, 0.0]], expected=True),
                dict(n=2, vec=[[0.0, -theta], [theta, 1.0]], expected=False),
                dict(
                    n=2,
                    vec=[[[0.0, -theta], [theta, 0.0]], [[0.0, -theta], [theta, 1.0]]],
                    expected=[True, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def is_tangent_compose_data(self):
            point = self.group.random_uniform()
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

        def to_tangent_data(self):
            theta = 1.0
            smoke_data = [
                dict(
                    n=2,
                    vec=[[0.0, -theta], [theta, 0.0]],
                    expected=[[0.0, -theta], [theta, 0.0]],
                )
            ]
            return self.generate_tests(smoke_data)

        def skew_to_vector_and_vector_to_skew_data(self):
            n_list = random.sample(range(2, 50), 10)
            random_data = [
                dict(n=n, mat=gs.random.rand(SpecialOrthogonal(n).dim)) for n in n_list
            ]
            return self.generate_tests([], random_data)

        def are_antipodals_data(self):
            mat1 = gs.eye(3)
            mat2 = gs.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
            smoke_data = [
                dict(n=3, mat1=mat1, mat2=mat2, expected=True),
                dict(n=3, mat1=[mat1, mat2], mat2=[mat2, mat2], expected=[True, False]),
            ]
            return self.generate_tests(smoke_data)

        def log_at_antipodals_value_error_data(self):
            smoke_data = [
                dict(
                    n=3,
                    mat1=gs.eye(3),
                    mat2=[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                    expected=pytest.raises(ValueError),
                ),
                dict(
                    n=3,
                    mat1=SpecialOrthogonal(3).random_uniform(),
                    mat2=SpecialOrthogonal(3).random_uniform(),
                    expected=does_not_raise(),
                ),
            ]
            return self.generate_tests(smoke_data)

        def from_vector_from_matrix_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples_list = random.sample(range(1, 20), 10)
            random_data = [
                dict(n=n, n_samples=n_samples)
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

        def rotation_vector_from_matrix_data(self):
            angle = 0.12
            smoke_data = [
                dict(
                    n=2,
                    mat=[
                        [1.0, 0.0, 0.0],
                        [0.0, gs.cos(angle), -gs.sin(angle)],
                        [0, gs.sin(angle), gs.cos(angle)],
                    ],
                    expected=0.12 * gs.array([1.0, 0.0, 0.0]),
                )
            ]
            return self.generate_tests(smoke_data)

        def distance_broadcast_shape_data(self):
            n_list = [2, 3]
            n_samples_list = random.sample(range(1, 20), 2)
            smoke_data = [
                dict(n=n, n_samples=n_samples)
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests(smoke_data)

        def projection_data(self):
            n_list = [2, 3]
            smoke_data = [
                dict(n=n, point_type="vector", mat=gs.eye(n) + 1e-12 * gs.ones((n, n)))
                for n in n_list
            ]
            return self.generate_tests(smoke_data)

        def projection_shape_data(self):
            n_list = [2, 3]
            n_samples_list = random.sample(range(1, 20), 2)
            random_data = [
                dict(n=n, n_samples=n_samples, expected=(n_samples, n, n))
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

        def skew_matrix_from_vector_data(self):
            smoke_data = dict(n=2, mat=[0.9], expected=[[0.0, -0.9], [0.9, 0.0]])
            return self.generate_tests(smoke_data)

        def log_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point=[2 * gs.pi / 5],
                    base_point=[gs.pi / 5],
                    expected=[1 * gs.pi / 5],
                )
            ]
            return self.generate_tests(smoke_data)

        def log_shape_data(self):
            random_data = []
            return self.generate_tests(random_data)

        def rotation_vector_rotation_matrix_regularize_data(self):
            n_list = random.sample(range(2, 50), 10)
            random_data = [
                dict(
                    n=n,
                    point=SpecialOrthogonal(n=3, point_type="vector").random_point(),
                )
                for n in n_list
            ]
            return self.generate_tests([], random_data)

        def parallel_transport_data(self):
            n_list = random.sample(range(2, 10), 5)
            n_samples_list = random.sample(range(2, 10), 5)
            random_data = [
                dict(n=n, n_samples=n_samples)
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

        def metric_left_invariant_data(self):

            smoke_data = [dict(n=3)]
            return self.generate_tests(smoke_data)

    testing_data = TestDataSpecialOrthogonal()

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(self.cls(n).belongs(mat), gs.array(expected))

    def test_dim(self, n, expected):
        self.assertAllClose(self.cls(n).dim, expected)

    def test_identity(self, n, expected):
        self.assertAllClose(self.cls(n).identity, gs.array(expected))

    def test_random_point_belongs(self, n, point_type, n_samples):
        group = self.cls(n)
        self.assertAllClose(gs.all(group(n).random_point(n_samples)), gs.array(True))

    def test_is_tangent(self, n, vec, expected):
        group = self.cls(n)
        self.assertAllClose(group.is_tangent(vec), gs.array(expected))

    def test_is_tangent_compose(self, n, point, vec, expected):
        group = self.cls(n)
        self.assertAllClose(group.compose(point, gs.array(vec)), gs.array(expected))

    def test_skew_to_vector_and_vector_to_skew(self, n, point_type, vec):
        group = self.cls(n, point_type)
        mat = group.skew_matrix_from_vector(gs.array(vec))
        result = group.vector_from_skew_matrix(mat)
        self.assertAllClose(result, vec)

    def test_are_antipodals(self, n, mat1, mat2, expected):
        group = self.cls(n)
        self.assertAllClose(group.are_antipodals(mat1, mat2), gs.array(expected))

    def test_log_at_antipodals_value_error(self, n, vec, expected):
        group = self.cls(n)
        with expected:
            group.log(vec)

    def test_from_vector_from_matrix(self, n, n_samples):
        group = self.cls(n)
        groupvec = self.cls(n, point_type="vector")
        point = groupvec.random_point(n_samples)
        rot_mat = group.matrix_from_rotation_vector(point)
        self.assertAllClose(
            group.rotation_vector_from_matrix(rot_mat), group.regularize(point)
        )

    def test_rotation_vector_from_matrix(self, n, mat, expected):
        group = self.cls(n)
        self.assertAllClose(
            group.rotation_vector_from_matrix(gs.array(mat)), gs.array(expected)
        )

    def test_distance_broadcast_shape(self, n, mat1, mat2, expected):
        group = self.cls(n)
        result = gs.shape(group.bi_invariant_metric.dist_broadcast(mat1, mat2))
        self.assertAllClose(result, expected)

    def test_projection(self, n, mat, expected):
        group = self.cls(n=n, point_type="vector")
        self.assertAllClose(group.projection(mat), expected)

    def test_projection_shape(self, n, vec, expected):
        group = self.cls(n=n, point_type="vector")
        self.assertAllClose(gs.shape(group.projection(gs.array(vec))), expected)

    def test_skew_matrix_from_vector(self, n, vec, expected):
        group = self.cls(n=n, point_type="vector")
        self.assertAllClose(group.skew_matrix_from_vector(gs.array(vec)), expected)

    def test_log(self, n, point, base_point, expected):
        group = self.cls(n=n, point_type="vector")
        log = group.log(gs.array(point), gs.array(base_point))
        self.assertAllClose(log, gs.array(expected))

    def test_rotation_vector_rotation_matrix_regularize(self, n, point):
        group = SpecialOrthogonal(n=n)
        rot_mat = group.matrix_from_rotation_vector(gs.array(point))
        self.assertAllClose(
            group.regularize(gs.array(point)),
            group.rotation_vector_from_matrix(rot_mat),
        )

    def test_parallel_transport(self, n, n_samples):
        metric = self.cls(n).bi_invariant_metric
        shape = (self.n_samples, self.group.n, self.group.n)
        result = gs.all(helper.test_parallel_transport(self.group, metric, shape))
        self.assertAllClose(result, gs.array(True))

    def test_metric_left_invariant(self, n):
        group = self.cls(n)
        point = group.random_point()
        tangent_vec = group.lie_algebra.basis[0]
        expected = group.bi_invariant_metric.norm(tangent_vec)

        translated = group.tangent_translation_map(point)(tangent_vec)
        result = group.bi_invariant_metric.norm(translated)
        self.assertAllClose(result, expected)


class TestSpecialOrthogonal3(TestCase, metaclass=Parametrizer):
    cls = SpecialOrthogonal

    class TestDataSpecialOrthogonal3(TestData):
        def tait_bryan_angles_matrix_data(self):
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
            for coord, order in itertools.product(coords, orders):
                for i in range(3):
                    vec = gs.zeros(3)
                    vec[i] = angle_pi_6
                    smoke_data += [
                        dict(coord=coord, order=order, vec=vec, mat=data[order][i])
                    ]
                smoke_data += [
                    dict(coord=coord, order=order, vec=gs.zeros(3), mat=gs.eye(3))
                ]
            return self.generate_tests(smoke_data)

        def tait_bryan_angles_quaternion_data(self):
            xyz = gs.array(
                [cos_angle_pi_12, sin_angle_pi_6, 0.0, 0.0],
                [cos_angle_pi_12, 0.0, sin_angle_pi_6, 0.0],
                [cos_angle_pi_12, 0.0, 0.0, sin_angle_pi_6],
            )

            zyx = gs.flip(xyz, axis=0)
            data = {"xyz": xyz, "zyx": zyx}
            smoke_data = []
            e1 = gs.array([1.0, 0.0, 0.0, 0.0])
            for coord, order in itertools.product(coords, orders):
                for i in range(3):
                    vec = gs.zeros(3)
                    vec[i] = angle_pi_6
                    smoke_data += [
                        dict(coord=coord, order=order, vec=vec, quat=data[order][i])
                    ]
                smoke_data += [dict(coord=coord, order=order, vec=gs.zeros(3), quat=e1)]
            return self.generate_tests(smoke_data)

        def quaternion_from_rotation_vector_tait_bryan_angles_data(self):
            smoke_data = []
            for coord, order in itertools.product(coords, orders):
                for angle_type in self.elements:
                    point = self.elements[angle_type]
                    if angle_type not in self.angles_close_to_pi:
                        smoke_data += [dict(coord=coord, order=order, point=point)]

            return self.generate_tests(smoke_data)

        def tait_bryan_angles_rotation_vector_data(self):
            smoke_data = []
            for coord, order in itertools.product(coords, orders):
                for angle_type in self.elements:
                    point = self.elements[angle_type]
                    if angle_type not in self.angles_close_to_pi:
                        smoke_data += [dict(coord=coord, order=order, point=point)]

            return self.generate_tests(smoke_data)

        def quaternion_and_rotation_vector_with_angles_close_to_pi_data(self):
            smoke_data = []
            angle_types = self.angles_close_to_pi
            for angle_type in angle_types:
                point = self.elements[angle_type]
                smoke_data += [dict(point=point)]

            return self.generate_tests(smoke_data)

        def quaternion_and_matrix_with_angles_close_to_pi_data(self):
            smoke_data = []
            angle_types = self.angles_close_to_pi
            for angle_type in angle_types:
                point = self.elements[angle_type]
                smoke_data += [dict(point=point)]

            return self.generate_tests(smoke_data)

        def rotation_vector_and_rotation_matrix_with_angles_close_to_pi_data(self):
            smoke_data = []
            angle_types = self.angles_close_to_pi
            for angle_type in angle_types:
                point = self.elements[angle_type]
                smoke_data += [dict(point=point)]

            return self.generate_tests(smoke_data)

    testing_data = TestDataSpecialOrthogonal3()

    def test_tait_bryan_angles_matrix(self, coord, order, vec, mat):
        group = self.cls(3, point_type="vector")

        mat_from_vec = group.matrix_from_tait_bryan_angles(vec, coord, order)
        self.assertAllClose(mat_from_vec, mat)
        vec_from_mat = group.tait_bryan_angles_from_matrix(mat, coord, order)
        self.assertAllClose(vec_from_mat, vec)

    def test_tait_bryan_angles_quaternion(self, coord, order, vec, quat):
        group = self.cls(3, point_type="vector")

        quat_from_vec = group.quaternion_from_tait_bryan_angles(vec, coord, order)
        self.assertAllClose(quat_from_vec, quat)
        vec_from_quat = group.tait_bryan_angles_from_matrix(quat, coord, order)
        self.assertAllClose(vec_from_quat, vec)

    def test_quaternion_from_rotation_vector_tait_bryan_angles(
        self, coord, order, point
    ):
        group = self.cls(3, point_type="vector")

        quat = group.quaternion_from_rotation_vector(point)
        tait_bryan_angle = group.tait_bryan_angles_from_quaternion(quat, coord, order)
        result = group.quaternion_from_tait_bryan_angles(tait_bryan_angle, coord, order)
        self.assertAllClose(result, quat)

    def test_tait_bryan_angles_rotation_vector(self, coord, order, point):
        group = self.cls(3, point_type="vector")

        tait_bryan_angle = group.tait_bryan_angles_from_rotation_vector(
            point, coord, order
        )
        result = group.rotation_vector_from_tait_bryan_angles(tait_bryan_angle)
        expected = group.regularize(point)
        self.assertAllClose(result, expected)

    def test_quaternion_and_rotation_vector_with_angles_close_to_pi(self, point):
        group = self.cls(3, point_type="vector")

        quaternion = group.quaternion_from_rotation_vector(point)
        result = group.rotation_vector_from_quaternion(quaternion)
        expected1 = group.regularize(point)
        expected2 = -1 * expected1
        expected = gs.allclose(result, expected1) or gs.allclose(result, expected2)
        self.assertAllClose(expected, gs.array(True))

    def test_quaternion_and_matrix_with_angles_close_to_pi(self, point):
        group = self.cls(3, point_type="vector")
        mat = group.matrix_from_rotation_vector(point)
        quat = group.quaternion_from_matrix(mat)
        result = group.matrix_from_quaternion(quat)
        expected1 = mat
        expected2 = gs.linalg.inv(mat)
        expected = gs.allclose(result, expected1) or gs.allclose(result, expected2)
        self.assertAllClose(expected, gs.array(True))

    def test_rotation_vector_and_rotation_matrix_with_angles_close_to_pi(self, point):
        group = self.cls(3, point_type="vector")
        mat = group.matrix_from_rotation_vector(point)
        result = group.rotation_vector_from_matrix(mat)
        expected1 = group.regularize(point)
        expected2 = -1 * expected1
        expected = gs.allclose(result, expected1) or gs.allclose(result, expected2)
        self.assertAllClose(expected, gs.array(True))
