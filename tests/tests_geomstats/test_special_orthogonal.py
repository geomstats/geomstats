"""Unit tests for special orthogonal group SO(n)."""

import pytest

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class TestSpecialOrthogonal(geomstats.tests.TestCase):
    def setup_method(self):
        self.n = 2
        self.group = SpecialOrthogonal(n=self.n)
        self.so3 = SpecialOrthogonal(n=3)
        self.n_samples = 4

    def test_dim(self):
        for n in [2, 3, 4, 5, 6]:
            group = SpecialOrthogonal(n=n)
            result = group.dim
            expected = n * (n - 1) / 2
            self.assertAllClose(result, expected)

    def test_belongs(self):
        theta = gs.pi / 3
        point_1 = gs.array(
            [[gs.cos(theta), -gs.sin(theta)], [gs.sin(theta), gs.cos(theta)]]
        )
        result = self.group.belongs(point_1)
        self.assertTrue(result)

        point_2 = gs.array(
            [[gs.cos(theta), gs.sin(theta)], [gs.sin(theta), gs.cos(theta)]]
        )
        result = self.group.belongs(point_2)
        self.assertFalse(result)

        point = gs.array([point_1, point_2])
        expected = gs.array([True, False])
        result = self.group.belongs(point)
        self.assertAllClose(result, expected)

        point = point_1[0]
        result = self.group.belongs(point)
        self.assertFalse(result)

        point = gs.zeros((2, 3))
        result = self.group.belongs(point)
        self.assertFalse(result)

        point = gs.zeros((2, 2, 3))
        result = self.group.belongs(point)
        self.assertFalse(gs.all(result))

    def test_random_uniform_and_belongs(self):
        point = self.group.random_uniform()
        result = self.group.belongs(point)
        expected = True
        self.assertAllClose(result, expected)

        point = self.group.random_uniform(self.n_samples)
        result = self.group.belongs(point)
        expected = gs.array([True] * self.n_samples)
        self.assertAllClose(result, expected)

    def test_identity(self):
        result = self.group.identity
        expected = gs.eye(self.n)
        self.assertAllClose(result, expected)

    def test_is_in_lie_algebra(self):
        theta = gs.pi / 3
        vec_1 = gs.array([[0.0, -theta], [theta, 0.0]])
        result = self.group.is_tangent(vec_1)
        self.assertTrue(result)

        vec_2 = gs.array([[0.0, -theta], [theta, 1.0]])
        result = self.group.is_tangent(vec_2)
        self.assertFalse(result)

        vec = gs.array([vec_1, vec_2])
        result = self.group.is_tangent(vec)
        expected = gs.array([True, False])
        self.assertAllClose(result, expected)

    def test_is_tangent(self):
        point = self.group.random_uniform()
        theta = 1.0
        vec_1 = gs.array([[0.0, -theta], [theta, 0.0]])
        vec_1 = self.group.compose(point, vec_1)
        result = self.group.is_tangent(vec_1, point)
        self.assertTrue(result)

        vec_2 = gs.array([[0.0, -theta], [theta, 1.0]])
        vec_2 = self.group.compose(point, vec_2)
        result = self.group.is_tangent(vec_2, point)
        self.assertFalse(result)

        vec = gs.array([vec_1, vec_2])
        point = gs.array([point, point])
        expected = gs.array([True, False])
        result = self.group.is_tangent(vec, point)
        self.assertAllClose(result, expected)

    def test_to_tangent(self):
        theta = 1.0
        vec_1 = gs.array([[0.0, -theta], [theta, 0.0]])
        result = self.group.to_tangent(vec_1)
        expected = vec_1
        self.assertAllClose(result, expected)

        n_samples = self.n_samples
        base_points = self.group.random_uniform(n_samples=n_samples)
        tangent_vecs = self.group.compose(base_points, vec_1)
        result = self.group.to_tangent(tangent_vecs, base_points)
        expected = tangent_vecs
        self.assertAllClose(result, expected)

    def test_projection_and_belongs(self):
        gs.random.seed(4)
        shape = (self.n_samples, self.n, self.n)
        result = helper.test_projection_and_belongs(self.group, shape, gs.atol * 100)
        for res in result:
            self.assertTrue(res)

    def test_skew_to_vec_and_back(self):
        group = SpecialOrthogonal(n=4)
        vec = gs.random.rand(group.dim)
        mat = group.skew_matrix_from_vector(vec)
        result = group.vector_from_skew_matrix(mat)
        self.assertAllClose(result, vec)

    def test_parallel_transport(self):
        metric = self.group.bi_invariant_metric
        shape = (self.n_samples, self.group.n, self.group.n)

        results = helper.test_parallel_transport(self.group, metric, shape)
        for res in results:
            self.assertTrue(res)

    def test_metric_left_invariant(self):
        group = self.group
        point = group.random_point()
        tangent_vec = self.group.lie_algebra.basis[0]
        expected = group.bi_invariant_metric.norm(tangent_vec)

        translated = group.tangent_translation_map(point)(tangent_vec)
        result = group.bi_invariant_metric.norm(translated)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_tf_only
    def test_distance_broadcast(self):
        group = self.group
        point = group.random_point(5)
        result = group.bi_invariant_metric.dist_broadcast(point[:3], point)
        expected = []
        for a in point[:3]:
            expected.append(group.bi_invariant_metric.dist(a, point))
        expected = gs.stack(expected)
        self.assertAllClose(result, expected)

    def test_are_antipodals(self):
        rotation_mat1 = gs.eye(3)
        rotation_mat2 = gs.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        result = self.so3.are_antipodals(rotation_mat1, rotation_mat2)

        self.assertTrue(result)

        rotvec1 = (2.0 * gs.pi / (3.0 * gs.sqrt(3.0))) * gs.array([1.0, 1.0, 1.0])
        rotvec2 = (-gs.pi / (3.0 * gs.sqrt(3.0))) * gs.array([1.0, 1.0, 1.0])
        rotation_mat1 = self.so3.matrix_from_rotation_vector(rotvec1)
        rotation_mat2 = self.so3.matrix_from_rotation_vector(rotvec2)

        result = self.so3.are_antipodals(rotation_mat1, rotation_mat2)
        self.assertTrue(gs.all(result))

        rotation_mat1 = self.so3.random_uniform()
        rotation_mat2 = rotation_mat1
        result = self.so3.are_antipodals(rotation_mat1, rotation_mat2)
        self.assertFalse(gs.all(result))

    def test_are_antipodals_vectorization(self):
        rotation_mat1 = gs.eye(3)
        rotation_mat2 = gs.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        rotation_mats1 = gs.array([rotation_mat1, rotation_mat2])
        rotation_mats2 = gs.array([rotation_mat2, rotation_mat2])
        result = self.so3.are_antipodals(rotation_mats1, rotation_mats2)
        expected = gs.array([True, False])
        self.assertAllClose(result, expected)

    def test_log_antipodals(self):
        rotation_mat1 = gs.eye(3)
        rotation_mat2 = gs.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        with pytest.raises(ValueError):
            self.so3.log(rotation_mat1, rotation_mat2)

    def test_matrix_from_rotation_vector(self):
        group = SpecialOrthogonal(n=3)
        rot_vec_0 = gs.array([0.0, 0.0, 0.0])
        result = group.matrix_from_rotation_vector(rot_vec_0)
        expected = gs.eye(3)
        self.assertAllClose(result, expected)

        rot_vec_1 = gs.array([gs.pi / 3.0, 0.0, 0.0])
        result = group.matrix_from_rotation_vector(rot_vec_1)
        expected = gs.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.5, -gs.sqrt(3.0) / 2],
                [0.0, gs.sqrt(3.0) / 2, 0.5],
            ]
        )
        self.assertAllClose(result, expected)

        rot_vec_3 = 1e-11 * gs.array([12.0, 1.0, -81.0])
        angle = gs.linalg.norm(rot_vec_3)
        skew_rot_vec_3 = 1e-11 * gs.array(
            [[0.0, 81.0, 1.0], [-81.0, 0.0, -12.0], [-1.0, 12.0, 0.0]]
        )
        coef_1 = gs.sin(angle) / angle
        coef_2 = (1.0 - gs.cos(angle)) / (angle**2)
        expected = (
            gs.eye(3)
            + coef_1 * skew_rot_vec_3
            + coef_2 * gs.matmul(skew_rot_vec_3, skew_rot_vec_3)
        )
        result = group.matrix_from_rotation_vector(rot_vec_3)
        self.assertAllClose(result, expected)

        rot_vec_6 = gs.array([0.1, 1.3, -0.5])
        angle = gs.linalg.norm(rot_vec_6)
        skew_rot_vec_6 = gs.array(
            [[0.0, 0.5, 1.3], [-0.5, 0.0, -0.1], [-1.3, 0.1, 0.0]]
        )

        coef_1 = gs.sin(angle) / angle
        coef_2 = (1 - gs.cos(angle)) / (angle**2)
        result = group.matrix_from_rotation_vector(rot_vec_6)
        expected = (
            gs.eye(3)
            + coef_1 * skew_rot_vec_6
            + coef_2 * gs.matmul(skew_rot_vec_6, skew_rot_vec_6)
        )
        self.assertAllClose(result, expected)

    def test_matrix_from_rotation_vector_vectorization(self):
        n_samples = self.n_samples
        group_vec = SpecialOrthogonal(n=3, point_type="vector")
        group_mat = SpecialOrthogonal(n=3)
        rot_vecs = group_vec.random_uniform(n_samples=n_samples)

        rot_mats = group_mat.matrix_from_rotation_vector(rot_vecs)

        self.assertAllClose(gs.shape(rot_mats), (n_samples, group_mat.n, group_mat.n))

    def test_rotation_vector_from_matrix(self):
        group = SpecialOrthogonal(n=3)

        angle = 0.12
        rot_mat = gs.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, gs.cos(angle), -gs.sin(angle)],
                [0, gs.sin(angle), gs.cos(angle)],
            ]
        )
        result = group.rotation_vector_from_matrix(rot_mat)
        expected = 0.12 * gs.array([1.0, 0.0, 0.0])

        self.assertAllClose(result, expected)

    def test_rotation_vector_and_rotation_matrix(self):
        """
        This tests that the composition of
        rotation_vector_from_matrix
        and
        matrix_from_rotation_vector
        is the identity.
        """
        group = SpecialOrthogonal(n=3)
        groupvec = SpecialOrthogonal(n=3, point_type="vector")
        point = groupvec.random_point()
        rot_mat = group.matrix_from_rotation_vector(point)
        result = group.rotation_vector_from_matrix(rot_mat)

        expected = self.group.regularize(point)

        self.assertAllClose(result, expected)
