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
