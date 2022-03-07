import itertools
import random
from contextlib import nullcontext as does_not_raise

import pytest

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.invariant_metric import BiInvariantMetric, InvariantMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.conftest import TestCase, tf_backend
from tests.data_generation import LieGroupTestData, RiemannianMetricTestData, TestData
from tests.parametrizers import (
    LieGroupParametrizer,
    Parametrizer,
    RiemannianMetricParametrizer,
)

EPSILON = 1e-5


def sample_matrix(theta, mul=1.0):
    return gs.array(
        [[gs.cos(theta), mul * gs.sin(theta)], [gs.sin(theta), gs.cos(theta)]]
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


elements = elements_all
if tf_backend():
    # Tf is extremely slow
    elements = {
        "with_angle_in_pi_2pi": with_angle_in_pi_2pi,
        "with_angle_close_pi_low": with_angle_close_pi_low,
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


class TestSpecialOrthogonal(TestCase, metaclass=LieGroupParametrizer):
    space = group = SpecialOrthogonal

    class TestDataSpecialOrthogonal(LieGroupTestData):
        n_list = random.sample(range(2, 5), 2)
        space_args_list = list(zip(n_list)) + [(2, "vector"), (3, "vector")]
        shape_list = [(n, n) for n in n_list] + [(1,), (3,)]
        n_samples_list = random.sample(range(2, 10), 4)
        n_points_list = random.sample(range(2, 10), 4)
        n_vecs_list = random.sample(range(2, 10), 4)

        def belongs_data(self):
            theta = gs.pi / 3
            smoke_data = [
                dict(n=2, mat=sample_matrix(theta, mul=-1.0), expected=True),
                dict(n=2, mat=sample_matrix(theta, mul=1.0), expected=False),
                dict(n=2, mat=gs.zeros((2, 3)), expected=False),
                dict(n=3, mat=gs.zeros((2, 3)), expected=False),
                dict(n=2, mat=gs.zeros((2, 2, 3)), expected=False),
                dict(
                    n=2,
                    mat=[
                        sample_matrix(theta / 2, mul=-1.0),
                        sample_matrix(theta / 2, mul=1.0),
                    ],
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

        def is_tangent_data(self):
            theta = gs.pi / 3
            point = SpecialOrthogonal(2).random_uniform()
            vec_1 = gs.array([[0.0, -theta], [theta, 0.0]])
            vec_2 = gs.array([[0.0, -theta], [theta, 1.0]])
            smoke_data = [
                dict(n=2, vec=vec_1, base_point=None, expected=True),
                dict(n=2, vec=vec_2, base_point=None, expected=False),
                dict(n=2, vec=[vec_1, vec_2], base_point=None, expected=[True, False]),
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

        def is_tangent_compose_data(self):
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
            random_data = []
            random_data += [
                dict(
                    n=2,
                    point_type="vector",
                    mat=SpecialOrthogonal(2, "vector").random_point(),
                )
            ]
            random_data += [
                dict(
                    n=3,
                    point_type="vector",
                    mat=SpecialOrthogonal(3, "vector").random_point(),
                )
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
                    point=gs.eye(3),
                    base_point=[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
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
            n_list = [2, 3]
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
                    n=3,
                    point_type="vector",
                    point=[
                        [1.0, 0.0, 0.0],
                        [0.0, gs.cos(angle), -gs.sin(angle)],
                        [0, gs.sin(angle), gs.cos(angle)],
                    ],
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
                dict(
                    n=n,
                    point_type="vector",
                    mat=gs.eye(n) + 1e-12 * gs.ones((n, n)),
                    expected=gs.eye(n),
                )
                for n in n_list
            ]
            return self.generate_tests(smoke_data)

        def projection_shape_data(self):
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

        def skew_matrix_from_vector_data(self):
            smoke_data = [dict(n=2, mat=[0.9], expected=[[0.0, -0.9], [0.9, 0.0]])]
            return self.generate_tests(smoke_data)

        def rotation_vector_rotation_matrix_regularize_data(self):
            n_list = [2, 3]
            random_data = [
                dict(
                    n=n,
                    point=SpecialOrthogonal(n=n, point_type="vector").random_point(),
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

        def matrix_from_rotation_vector_data(self):

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
                dict(dim=3, rot_vec=gs.array([0.0, 0.0, 0.0]), expected=gs.eye(3)),
                dict(
                    dim=3,
                    rot_vec=gs.array([gs.pi / 3.0, 0.0, 0.0]),
                    expected=gs.array(
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 0.5, -gs.sqrt(3.0) / 2],
                            [0.0, gs.sqrt(3.0) / 2, 0.5],
                        ]
                    ),
                ),
                dict(dim=3, rot_vec=rot_vec_3, expected=expected_3),
                dict(dim=3, rot_vec=rot_vec_4, expected=expected_4),
                dict(
                    dim=2,
                    rot_vec=gs.array([gs.pi / 3]),
                    expected=gs.array(
                        [[1.0 / 2, -gs.sqrt(3.0) / 2], [gs.sqrt(3.0) / 2, 1.0 / 2]]
                    ),
                ),
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_data(self):
            smoke_space_args_list = [(2, True), (3, True), (2, False)]
            smoke_n_points_list = [1, 2, 1]
            return self._random_point_belongs_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_data(self):
            space_args_list = list(zip(self.n_list))
            shape_list = [(n, n) for n in self.n_list]
            n_samples_list = random.sample(range(2, 10), 2)
            return self._projection_belongs_data(
                space_args_list, shape_list, n_samples_list, gs.atol * 1000
            )

        def to_tangent_is_tangent_data(self):
            space_args_list = list(zip(self.n_list))
            shape_list = [(n, n) for n in self.n_list]
            n_vecs_list = random.sample(range(2, 10), 2)
            return self._to_tangent_is_tangent_data(
                SpecialOrthogonal,
                space_args_list,
                shape_list,
                n_vecs_list,
            )

        def exp_log_composition_data(self):
            return self._exp_log_composition_data(
                SpecialOrthogonal,
                self.space_args_list,
                self.shape_list,
                self.n_samples_list,
            )

        def log_exp_composition_data(self):
            return self._log_exp_composition_data(
                SpecialOrthogonal, self.space_args_list, self.n_samples_list
            )

        def compose_point_inverse_is_identity_data(self):
            smoke_data = []
            for space_args in self.space_args_list:
                smoke_data += [dict(space_args=space_args)]
            return self.generate_tests(smoke_data)

        def compose_data(self):
            smoke_data = [
                dict(
                    dim=2,
                    point_type="vector",
                    point_a=gs.array([0.12]),
                    point_b=gs.array([-0.15]),
                    expected=gs.array([-0.03]),
                )
            ]
            return self.generate_tests(smoke_data)

        def log_data(self):
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

        def exp_data(self):
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

        def compose_shape_data(self):
            smoke_data = [
                dict(n=3, point_type="vector", n_samples=4),
                dict(n=2, point_type="matrix", n_samples=4),
                dict(n=3, point_type="matrix", n_samples=4),
                dict(n=4, point_type="matrix", n_samples=4),
            ]
            return self.generate_tests(smoke_data)

        def rotation_vector_and_rotation_matrix_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point_type="vector",
                    rot_vec=gs.array([[2.0], [1.3], [0.8], [0.03]]),
                ),
                dict(n=2, point_type="vector", rot_vec=gs.array([0.78])),
            ]
            return self.generate_tests(smoke_data)

        def regularize_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point_type="vector",
                    angle=gs.array([2 * gs.pi + 1]),
                    expected=gs.array([1.0]),
                )
            ]
            return self.generate_tests(smoke_data)

    testing_data = TestDataSpecialOrthogonal()

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(self.space(n).belongs(gs.array(mat)), gs.array(expected))

    def test_dim(self, n, expected):
        self.assertAllClose(self.space(n).dim, expected)

    def test_identity(self, n, expected):
        self.assertAllClose(self.space(n).identity, gs.array(expected))

    def test_is_tangent(self, n, vec, base_point, expected):
        group = self.space(n)
        self.assertAllClose(
            group.is_tangent(gs.array(vec), base_point), gs.array(expected)
        )

    def test_skew_to_vector_and_vector_to_skew(self, n, point_type, vec):
        group = self.space(n, point_type)
        mat = group.skew_matrix_from_vector(gs.array(vec))
        result = group.vector_from_skew_matrix(mat)
        self.assertAllClose(result, vec)

    def test_are_antipodals(self, n, mat1, mat2, expected):
        group = self.space(n)
        self.assertAllClose(group.are_antipodals(mat1, mat2), gs.array(expected))

    def test_log_at_antipodals_value_error(self, n, point, base_point, expected):
        group = self.space(n)
        with expected:
            group.log(point, base_point)

    def test_from_vector_from_matrix(self, n, n_samples):
        group = self.space(n)
        groupvec = self.space(n, point_type="vector")
        point = groupvec.random_point(n_samples)
        rot_mat = group.matrix_from_rotation_vector(point)
        self.assertAllClose(
            group.rotation_vector_from_matrix(rot_mat), group.regularize(point)
        )

    def test_rotation_vector_from_matrix(self, n, point_type, point, expected):
        group = self.space(n, point_type)
        self.assertAllClose(
            group.rotation_vector_from_matrix(gs.array(point)), gs.array(expected)
        )

    def test_projection(self, n, point_type, mat, expected):
        group = self.space(n=n, point_type=point_type)
        self.assertAllClose(group.projection(mat), expected)

    def test_projection_shape(self, n, point_type, n_samples, expected):
        group = self.space(n=n, point_type=point_type)
        self.assertAllClose(
            gs.shape(group.projection(group.random_point(n_samples))), expected
        )

    def test_skew_matrix_from_vector(self, n, vec, expected):
        group = self.space(n=n, point_type="vector")
        self.assertAllClose(group.skew_matrix_from_vector(gs.array(vec)), expected)

    def test_rotation_vector_rotation_matrix_regularize(self, n, point):
        group = SpecialOrthogonal(n=n)
        rot_mat = group.matrix_from_rotation_vector(gs.array(point))
        self.assertAllClose(
            group.regularize(gs.array(point)),
            group.rotation_vector_from_matrix(rot_mat),
        )

    def test_matrix_from_rotation_vector(self, dim, rot_vec, expected):
        group = SpecialOrthogonal(dim)
        result = group.matrix_from_rotation_vector(rot_vec)
        self.assertAllClose(result, expected)

    def test_compose_point_inverse_is_identity(self, space_args):
        group = SpecialOrthogonal(*space_args)
        point = gs.squeeze(group.random_point())
        inv_point = group.inverse(point)
        self.assertAllClose(group.compose(point, inv_point), group.identity)

    def test_compose(self, n, point_type, point_a, point_b, expected):
        group = SpecialOrthogonal(n, point_type)
        result = group.compose(point_a, point_b)
        expected = group.regularize(expected)
        self.assertAllClose(result, expected)

    def test_regularize(self, n, point_type, angle, expected):
        group = SpecialOrthogonal(n, point_type)
        result = group.regularize(angle)
        self.assertAllClose(result, expected)

    def test_exp(self, n, point_type, tangent_vec, base_point, expected):
        """
        The Riemannian exp and log are inverse functions of each other.
        This test is the inverse of test_log's.
        """
        group = self.space(n, point_type)
        result = group.exp(tangent_vec, base_point)
        self.assertAllClose(result, expected)

    def test_log(self, n, point_type, point, base_point, expected):
        """
        The Riemannian exp and log are inverse functions of each other.
        This test is the inverse of test_exp's.
        """
        group = self.space(n, point_type)
        result = group.log(point=point, base_point=base_point)
        self.assertAllClose(result, expected)

    def test_compose_shape(self, n, point_type, n_samples):
        group = self.space(n, point_type=point_type)
        n_points_a = group.random_uniform(n_samples=n_samples)
        n_points_b = group.random_uniform(n_samples=n_samples)
        one_point = group.random_uniform(n_samples=1)

        result = group.compose(one_point, n_points_a)
        self.assertAllClose(gs.shape(result), (n_samples,) + group.shape)

        result = group.compose(n_points_a, one_point)
        self.assertAllClose(gs.shape(result), (n_samples,) + group.shape)

        result = group.compose(n_points_a, n_points_b)
        self.assertAllClose(gs.shape(result), (n_samples,) + group.shape)

    def test_rotation_vector_and_rotation_matrix(self, n, point_type, rot_vec):
        group = self.space(n, point_type=point_type)
        rot_mats = group.matrix_from_rotation_vector(rot_vec)
        result = group.rotation_vector_from_matrix(rot_mats)
        expected = group.regularize(rot_vec)
        self.assertAllClose(result, expected)


class TestSpecialOrthogonal3Vectors(TestCase, metaclass=Parametrizer):
    space = group = SpecialOrthogonal

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
            for coord, order in itertools.product(["intrinsic", "extrinsic"], orders):
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
                for angle_type in elements_all:
                    point = elements_all[angle_type]
                    if angle_type not in angles_close_to_pi:
                        smoke_data += [dict(coord=coord, order=order, point=point)]

            return self.generate_tests(smoke_data)

        def tait_bryan_angles_rotation_vector_data(self):
            smoke_data = []
            for coord, order in itertools.product(coords, ["xyz"]):
                for angle_type in elements_all:
                    point = elements_all[angle_type]
                    if angle_type not in angles_close_to_pi:
                        smoke_data += [dict(coord=coord, order=order, point=point)]

            return self.generate_tests(smoke_data)

        def quaternion_and_rotation_vector_with_angles_close_to_pi_data(self):
            smoke_data = []
            angle_types = angles_close_to_pi
            for angle_type in angle_types:
                point = elements_all[angle_type]
                smoke_data += [dict(point=point)]

            return self.generate_tests(smoke_data)

        def quaternion_and_matrix_with_angles_close_to_pi_data(self):
            smoke_data = []
            angle_types = angles_close_to_pi
            for angle_type in angle_types:
                point = elements_all[angle_type]
                smoke_data += [dict(point=point)]

            return self.generate_tests(smoke_data)

        def rotation_vector_and_rotation_matrix_with_angles_close_to_pi_data(self):
            smoke_data = []
            angle_types = angles_close_to_pi
            for angle_type in angle_types:
                point = elements_all[angle_type]
                smoke_data += [dict(point=point)]

            return self.generate_tests(smoke_data)

        def lie_bracket_data(self):
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

        def group_log_then_exp_with_angles_close_to_pi_data(self):
            smoke_data = []
            for angle_type in angles_close_to_pi:
                for angle_type_base in elements:
                    smoke_data += [
                        dict(
                            point=elements[angle_type],
                            base_point=elements[angle_type_base],
                        )
                    ]
            return self.generate_tests(smoke_data)

        def group_exp_then_log_with_angles_close_to_pi_data(self):
            return self.group_log_then_exp_with_angles_close_to_pi_data()

        def left_jacobian_vectorization_data(self):
            smoke_data = [dict(n_samples=3)]
            return self.generate_tests(smoke_data)

        def left_jacobian_through_its_determinant_data(self):
            smoke_data = []
            for angle_type in elements:
                point = elements[angle_type]
                angle = gs.linalg.norm(SpecialOrthogonal(3, "vector").regularize(point))
                if angle_type in [
                    "with_angle_0",
                    "with_angle_close_0",
                    "with_angle_2pi",
                    "with_angle_close_2pi_high",
                ]:
                    expected = 1.0 + angle**2 / 12.0 + angle**4 / 240.0
                else:
                    expected = angle**2 / (4 * gs.sin(angle / 2) ** 2)
                smoke_data += [dict(point=point, expected=expected)]

            return self.generate_tests(smoke_data)

        def inverse_data(self):
            smoke_data = [dict(n_samples=3)]
            return self.generate_tests(smoke_data)

        def compose_and_inverse_data(self):
            smoke_data = []
            for point in elements.values():
                smoke_data += [dict(point=point)]
            return self.generate_tests(smoke_data)

        def compose_regularize_data(self):
            smoke_data = []
            for element_type in elements:
                point = elements[element_type]
                if element_type not in angles_close_to_pi:
                    smoke_data += [dict(point=point)]
            return self.generate_tests(smoke_data)

        def compose_regularize_angles_close_to_pi_data(self):
            smoke_data = []
            for element_type in elements:
                point = elements[element_type]
                if element_type in angles_close_to_pi:
                    smoke_data += [dict(point=point)]
            return self.generate_tests(smoke_data)

        def regularize_extreme_cases_data(self):
            smoke_data = [dict(elements_all=elements_all)]
            return self.generate_tests(smoke_data)

        def regularize_data(self):
            point = (gs.pi + 1e-6) * gs.array(
                [[1.0, 0.0, 0.0], [2, 0.5, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
            )
            expected_2 = (
                point[1]
                / gs.linalg.norm(point[1])
                * (gs.linalg.norm(point[1]) - 2 * gs.pi)
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

    testing_data = TestDataSpecialOrthogonal3()

    def test_tait_bryan_angles_matrix(self, coord, order, vec, mat):
        group = self.space(3, point_type="vector")

        mat_from_vec = group.matrix_from_tait_bryan_angles(vec, coord, order)
        self.assertAllClose(mat_from_vec, mat)
        vec_from_mat = group.tait_bryan_angles_from_matrix(mat, coord, order)
        self.assertAllClose(vec_from_mat, vec)

    def test_tait_bryan_angles_quaternion(self, coord, order, vec, quat):
        group = self.space(3, point_type="vector")

        quat_from_vec = group.quaternion_from_tait_bryan_angles(vec, coord, order)
        self.assertAllClose(quat_from_vec, quat)
        vec_from_quat = group.tait_bryan_angles_from_quaternion(quat, coord, order)
        self.assertAllClose(vec_from_quat, vec)

    def test_quaternion_from_rotation_vector_tait_bryan_angles(
        self, coord, order, point
    ):
        group = self.space(3, point_type="vector")

        quat = group.quaternion_from_rotation_vector(point)
        tait_bryan_angle = group.tait_bryan_angles_from_quaternion(quat, coord, order)
        result = group.quaternion_from_tait_bryan_angles(tait_bryan_angle, coord, order)
        self.assertAllClose(result, quat)

    def test_tait_bryan_angles_rotation_vector(self, coord, order, point):
        group = self.space(3, point_type="vector")

        tait_bryan_angle = group.tait_bryan_angles_from_rotation_vector(
            point, coord, order
        )
        result = group.rotation_vector_from_tait_bryan_angles(
            tait_bryan_angle, coord, order
        )
        expected = group.regularize(point)
        self.assertAllClose(result, expected)

    def test_quaternion_and_rotation_vector_with_angles_close_to_pi(self, point):
        group = self.space(3, point_type="vector")

        quaternion = group.quaternion_from_rotation_vector(point)
        result = group.rotation_vector_from_quaternion(quaternion)
        expected1 = group.regularize(point)
        expected2 = -1 * expected1
        expected = gs.allclose(result, expected1) or gs.allclose(result, expected2)
        self.assertAllClose(expected, gs.array(True))

    def test_quaternion_and_matrix_with_angles_close_to_pi(self, point):
        group = self.space(3, point_type="vector")
        mat = group.matrix_from_rotation_vector(point)
        quat = group.quaternion_from_matrix(mat)
        result = group.matrix_from_quaternion(quat)
        expected1 = mat
        expected2 = gs.linalg.inv(mat)
        expected = gs.allclose(result, expected1) or gs.allclose(result, expected2)
        self.assertAllClose(expected, gs.array(True))

    def test_rotation_vector_and_rotation_matrix_with_angles_close_to_pi(self, point):
        group = self.space(3, point_type="vector")
        mat = group.matrix_from_rotation_vector(point)
        result = group.rotation_vector_from_matrix(mat)
        expected1 = group.regularize(point)
        expected2 = -1 * expected1
        expected = gs.allclose(result, expected1) or gs.allclose(result, expected2)
        self.assertAllClose(expected, gs.array(True))

    def test_lie_bracket(self, tangent_vec_a, tangent_vec_b, base_point, expected):
        group = self.space(3, point_type="vector")
        result = group.lie_bracket(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_group_log_then_exp_with_angles_close_to_pi(self, point, base_point):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        # TODO(nguigs): fix this test for tf
        group = self.space(3, point_type="vector")
        result = group.exp(group.log(point, base_point), base_point)
        expected = group.regularize(point)
        inv_expected = -expected

        self.assertTrue(
            gs.allclose(result, expected, atol=5e-3)
            or gs.allclose(result, inv_expected, atol=5e-3)
        )

    def test_group_exp_then_log_with_angles_close_to_pi(self, tangent_vec, base_point):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        group = self.space(3, point_type="vector")
        result = group.log(group.exp(tangent_vec, base_point), base_point)
        metric = group.left_canonical_metric
        reg_tangent_vec = group.regularize_tangent_vec(
            tangent_vec=tangent_vec, base_point=base_point, metric=metric
        )
        expected = reg_tangent_vec
        inv_expected = -expected
        self.assertTrue(
            gs.allclose(result, expected, atol=5e-3)
            or gs.allclose(result, inv_expected, atol=5e-3)
        )

    def test_left_jacobian_vectorization(self, n_samples):
        group = self.space(3, point_type="vector")
        points = group.random_uniform(n_samples=n_samples)
        jacobians = group.jacobian_translation(point=points, left_or_right="left")
        self.assertAllClose(gs.shape(jacobians), (n_samples, group.dim, group.dim))

    def test_inverse(self, n_samples):
        group = self.space(3, point_type="vector")
        points = group.random_uniform(n_samples=n_samples)
        result = group.inverse(points)

        self.assertAllClose(gs.shape(result), (n_samples, group.dim))

    def test_left_jacobian_through_its_determinant(self, point, expected):
        group = self.space(3, point_type="vector")
        jacobian = group.jacobian_translation(point=point, left_or_right="left")
        result = gs.linalg.det(jacobian)
        self.assertAllClose(result, expected)

    def test_compose_and_inverse(self, point):
        group = self.space(3, point_type="vector")
        inv_point = group.inverse(point)
        result = group.compose(point, inv_point)
        expected = group.identity
        self.assertAllClose(result, expected)
        result = group.compose(inv_point, point)
        self.assertAllClose(result, expected)

    def test_compose_regularize(self, point):
        group = self.space(3, point_type="vector")
        result = group.compose(point, group.identity)
        expected = group.regularize(point)
        self.assertAllClose(result, expected)

        result = group.compose(group.identity, point)
        expected = group.regularize(point)
        self.assertAllClose(result, expected)

    def test_compose_regularize_angles_close_to_pi(self, point):
        group = self.space(3, point_type="vector")
        result = group.compose(point, group.identity)
        expected = group.regularize(point)
        inv_expected = -expected
        self.assertTrue(
            gs.allclose(result, expected) or gs.allclose(result, inv_expected)
        )

        result = group.compose(group.identity, point)
        expected = group.regularize(point)
        inv_expected = -expected
        self.assertTrue(
            gs.allclose(result, expected) or gs.allclose(result, inv_expected)
        )

    @geomstats.tests.np_autograd_and_tf_only
    def test_regularize_extreme_cases(self, elements_all):
        group = SpecialOrthogonal(3, "vector")
        point = elements_all["with_angle_0"]
        self.assertAllClose(gs.linalg.norm(point), 0.0)
        result = group.regularize(point)
        expected = point
        self.assertAllClose(result, expected)

        less_than_pi = ["with_angle_close_0", "with_angle_close_pi_low"]
        for angle_type in less_than_pi:
            point = elements_all[angle_type]
            result = group.regularize(point)
            expected = point
            self.assertAllClose(result, expected)

        angle_type = "with_angle_pi"
        point = elements_all[angle_type]
        result = group.regularize(point)
        expected = point
        self.assertAllClose(result, expected)

        angle_type = "with_angle_close_pi_high"
        point = elements_all[angle_type]
        result = group.regularize(point)
        self.assertTrue(0 <= gs.linalg.norm(result) < gs.pi)
        norm = gs.linalg.norm(point)
        expected = point / norm * (norm - 2 * gs.pi)
        self.assertAllClose(result, expected)

        in_pi_2pi = ["with_angle_in_pi_2pi", "with_angle_close_2pi_low"]

        for angle_type in in_pi_2pi:
            point = elements_all[angle_type]
            angle = gs.linalg.norm(point)
            new_angle = gs.pi - (angle - gs.pi)

            point_initial = point
            result = group.regularize(point)

            expected = -(new_angle / angle) * point_initial
            self.assertAllClose(result, expected)

        angle_type = "with_angle_2pi"
        point = elements_all[angle_type]
        result = group.regularize(point)
        expected = gs.array([0.0, 0.0, 0.0])
        self.assertAllClose(result, expected)

        angle_type = "with_angle_close_2pi_high"
        point = elements_all[angle_type]
        angle = gs.linalg.norm(point)
        new_angle = angle - 2 * gs.pi

        result = group.regularize(point)
        expected = new_angle * point / angle
        self.assertAllClose(result, expected)

    def test_regularize(self, point, expected):
        group = SpecialOrthogonal(3, "vector")
        result = group.regularize(point)
        self.assertAllClose(result, expected)


class TestBiInvariantMetric(TestCase, metaclass=RiemannianMetricParametrizer):
    metric = connection = BiInvariantMetric
    skip_test_exp_geodesic_ivp = True

    class TestDataBiInvariantMetric(RiemannianMetricTestData):
        dim_list = random.sample(range(2, 6), 4)
        metric_args_list = [(SpecialOrthogonal(dim),) for dim in dim_list]
        shape_list = [(dim, dim) for dim in dim_list]
        space_list = [SpecialOrthogonal(dim) for dim in dim_list]
        n_points_list = random.sample(range(1, 10), 4)
        n_samples_list = random.sample(range(1, 10), 4)
        n_points_a_list = random.sample(range(1, 10), 4)
        n_points_b_list = [1]
        batch_size_list = random.sample(range(2, 10), 4)
        alpha_list = [1] * 2
        n_rungs_list = [1] * 2
        scheme_list = ["pole"] * 2

        def exp_shape_data(self):
            return self._exp_shape_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.batch_size_list,
            )

        def log_shape_data(self):
            return self._log_shape_data(
                self.metric_args_list,
                self.space_list,
                self.batch_size_list,
            )

        def squared_dist_is_symmetric_data(self):
            return self._squared_dist_is_symmetric_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
                atol=gs.atol * 1000,
            )

        def exp_belongs_data(self):
            return self._exp_belongs_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_is_tangent_data(self):
            return self._log_is_tangent_data(
                self.metric_args_list,
                self.space_list,
                self.n_samples_list,
                is_tangent_atol=gs.atol * 1000,
            )

        def geodesic_ivp_belongs_data(self):
            return self._geodesic_ivp_belongs_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def geodesic_bvp_belongs_data(self):
            return self._geodesic_bvp_belongs_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_exp_composition_data(self):
            return self._log_exp_composition_data(
                self.metric_args_list,
                self.space_list,
                self.n_samples_list,
                rtol=gs.rtol * 1000,
                atol=gs.atol * 1000,
            )

        def exp_log_composition_data(self):
            return self._exp_log_composition_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                rtol=gs.rtol * 1000,
                atol=gs.atol * 1000,
            )

        def exp_ladder_parallel_transport_data(self):
            return self._exp_ladder_parallel_transport_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                self.n_rungs_list,
                self.alpha_list,
                self.scheme_list,
            )

        def exp_geodesic_ivp_data(self):
            return self._exp_geodesic_ivp_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                self.n_points_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 100,
            )

        def parallel_transport_ivp_is_isometry_data(self):
            return self._parallel_transport_ivp_is_isometry_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                is_tangent_atol=gs.atol * 100,
                rtol=gs.rtol * 100,
                atol=gs.atol * 100,
            )

        def parallel_transport_bvp_is_isometry_data(self):
            return self._parallel_transport_bvp_is_isometry_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
                is_tangent_atol=gs.atol * 100,
                rtol=gs.rtol * 100,
                atol=gs.atol * 100,
            )

        def log_exp_intrinsic_ball_extrinsic_data(self):
            smoke_data = [
                dict(
                    dim=2,
                    x_intrinsic=gs.array([4.0, 0.2]),
                    y_intrinsic=gs.array([3.0, 3]),
                )
            ]
            return self.generate_tests(smoke_data)

        def squared_dist_is_less_than_squared_pi_data(self):
            smoke_data = []
            for angle_type_1, angle_type_2 in zip(elements, elements):
                smoke_data += [
                    dict(point_1=elements[angle_type_1], point_2=elements[angle_type_2])
                ]
            return self.generate_tests(smoke_data)

        def exp_data(self):
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
            expected = SpecialOrthogonal(3, "vector").compose(
                (gs.pi / 5.0) / gs.sqrt(3.0) * gs.array([1.0, 1.0, 1.0]),
                gs.dot(inv_jacobian, rot_vec_2),
            )
            smoke_data = [
                dict(
                    tangent_vec=gs.array([0.0, 0.0, 0.0]),
                    base_point=rot_vec_base_point,
                    expected=rot_vec_base_point,
                ),
                dict(
                    tangent_vec=rot_vec_2,
                    base_point=rot_vec_base_point,
                    expected=expected,
                ),
            ]
            return self.generate_tests(smoke_data)

        def log_data(self):
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
            rot_vec_2 = SpecialOrthogonal(3, "vector").compose(rot_vec_base_point, aux)

            smoke_data = [
                dict(
                    point=rot_vec_base_point,
                    base_point=rot_vec_base_point,
                    expected=gs.array([0.0, 0.0, 0.0]),
                ),
                dict(
                    point=rot_vec_2,
                    base_point=rot_vec_base_point,
                    expected=expected,
                ),
            ]
            return self.generate_tests(smoke_data)

        def distance_broadcast_data(self):
            smoke_data = [dict(n=2)]
            return self.generate_tests(smoke_data)

    testing_data = TestDataBiInvariantMetric()

    def test_squared_dist_is_less_than_squared_pi(self, point_1, point_2):
        """
        This test only concerns the canonical metric.
        For other metrics, the scaling factor can give
        distances above pi.
        """
        group = SpecialOrthogonal(3, "vector")
        metric = self.metric(SpecialOrthogonal(3, "vector"))
        point_1 = group.regularize(point_1)
        point_2 = group.regularize(point_2)

        sq_dist = metric.squared_dist(point_1, point_2)
        diff = sq_dist - gs.pi**2
        self.assertTrue(
            diff <= 0 or abs(diff) < EPSILON, "sq_dist = {}".format(sq_dist)
        )

    def test_exp(self, tangent_vec, base_point, expected):
        metric = self.metric(SpecialOrthogonal(3, "vector"))
        result = metric.exp(tangent_vec, base_point)
        self.assertAllClose(result, expected)

    def test_log(self, point, base_point, expected):
        metric = self.metric(SpecialOrthogonal(3, "vector"))
        result = metric.log(point, base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_tf_only
    def test_distance_broadcast(self, n):
        group = SpecialOrthogonal(n=n)
        point = group.random_point(5)
        result = group.bi_invariant_metric.dist_broadcast(point[:3], point)
        expected = []
        for a in point[:3]:
            expected.append(group.bi_invariant_metric.dist(a, point))
        expected = gs.stack(expected)
        self.assertAllClose(result, expected)


class TestInvariantMetricOnSO3(TestCase, metaclass=Parametrizer):
    metric = connection = InvariantMetric
    skip_test_exp_geodesic_ivp = True

    class TestDataInvariantMetric(TestData):
        def squared_dist_is_symmetric_data(self):
            smoke_data = []
            for angle_type_1, angle_type_2, left_or_right in zip(
                elements, elements, ["left", "right"]
            ):
                smoke_data += [
                    dict(
                        metric_mat_at_identity=9
                        * gs.eye(SpecialOrthogonal(3, "vector").dim),
                        left_or_right=left_or_right,
                        point_1=elements[angle_type_1],
                        point_2=elements[angle_type_2],
                    )
                ]
            return self.generate_tests(smoke_data)

    testing_data = TestDataInvariantMetric()

    def test_squared_dist_is_symmetric(
        self, metric_mat_at_identity, left_or_right, point_1, point_2
    ):
        group = SpecialOrthogonal(3, "vector")
        metric = self.metric(
            SpecialOrthogonal(n=3, point_type="vector"),
            metric_mat_at_identity=metric_mat_at_identity,
            left_or_right=left_or_right,
        )
        point_1 = group.regularize(point_1)
        point_2 = group.regularize(point_2)

        sq_dist_1_2 = gs.mod(metric.squared_dist(point_1, point_2) + 1e-4, gs.pi**2)
        sq_dist_2_1 = gs.mod(metric.squared_dist(point_2, point_1) + 1e-4, gs.pi**2)
        self.assertAllClose(sq_dist_1_2, sq_dist_2_1, atol=1e-4)
