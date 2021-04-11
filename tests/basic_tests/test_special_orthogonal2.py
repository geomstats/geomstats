"""Unit tests for special orthogonal group SO(3)."""

import warnings

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats import algebra_utils
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


EPSILON = 1e-5
ATOL = 1e-5


class TestSpecialOrthogonal2(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=UserWarning)

        gs.random.seed(1234)

        self.group = SpecialOrthogonal(n=2, point_type='vector')

        # -- Set attributes
        self.n_samples = 4

    def test_projection(self):
        # Test 2D and nD cases
        rot_mat = gs.eye(2)
        delta = 1e-12 * gs.ones((2, 2))
        rot_mat_plus_delta = rot_mat + delta
        result = self.group.projection(rot_mat_plus_delta)
        expected = rot_mat
        self.assertAllClose(result, expected)

    def test_projection_vectorization(self):
        n_samples = self.n_samples
        mats = gs.ones((n_samples, 2, 2))
        result = self.group.projection(mats)
        self.assertAllClose(gs.shape(result), (n_samples, 2, 2))

    def test_skew_matrix_from_vector(self):
        rot_vec = gs.array([0.9])
        skew_matrix = self.group.skew_matrix_from_vector(rot_vec)
        result = gs.matmul(skew_matrix, skew_matrix)
        diag = gs.array([-0.81, -0.81])
        expected = algebra_utils.from_vector_to_diagonal_matrix(diag)
        self.assertAllClose(result, expected)

    def test_skew_matrix_and_vector(self):
        rot_vec = gs.array([0.8])

        skew_mat = self.group.skew_matrix_from_vector(rot_vec)
        result = self.group.vector_from_skew_matrix(skew_mat)
        expected = rot_vec

        self.assertAllClose(result, expected)

    def test_skew_matrix_from_vector_vectorization(self):
        n_samples = self.n_samples
        rot_vecs = self.group.random_uniform(n_samples=n_samples)
        result = self.group.skew_matrix_from_vector(rot_vecs)

        self.assertAllClose(gs.shape(result), (n_samples, 2, 2))

    def test_random_uniform_shape(self):
        result = self.group.random_uniform()
        self.assertAllClose(gs.shape(result), (self.group.dim,))

    def test_random_and_belongs(self):
        point = self.group.random_uniform()
        result = self.group.belongs(point)
        expected = True
        self.assertAllClose(result, expected)

    def test_random_and_belongs_vectorization(self):
        n_samples = self.n_samples
        points = self.group.random_uniform(n_samples=n_samples)
        result = self.group.belongs(points)
        expected = gs.array([True] * n_samples)
        self.assertAllClose(result, expected)

    def test_regularize(self):
        angle = 2 * gs.pi + 1
        result = self.group.regularize(gs.array([angle]))
        expected = gs.array([1.])
        self.assertAllClose(result, expected)

    def test_regularize_vectorization(self):
        n_samples = self.n_samples
        rot_vecs = self.group.random_uniform(n_samples=n_samples)
        result = self.group.regularize(rot_vecs)

        self.assertAllClose(gs.shape(result), (n_samples, self.group.dim))

    def test_matrix_from_rotation_vector(self):
        angle = gs.pi / 3
        expected = gs.array([[1. / 2, -gs.sqrt(3.) / 2],
                             [gs.sqrt(3.) / 2, 1. / 2]])
        result = self.group.matrix_from_rotation_vector(gs.array([angle]))
        self.assertAllClose(result, expected)

    def test_matrix_from_rotation_vector_vectorization(self):
        n_samples = self.n_samples
        rot_vecs = self.group.random_uniform(n_samples=n_samples)

        rot_mats = self.group.matrix_from_rotation_vector(rot_vecs)

        self.assertAllClose(
            gs.shape(rot_mats), (n_samples, self.group.n, self.group.n))

    def test_rotation_vector_from_matrix(self):
        angle = .12
        rot_mat = gs.array([[gs.cos(angle), -gs.sin(angle)],
                            [gs.sin(angle), gs.cos(angle)]])
        result = self.group.rotation_vector_from_matrix(rot_mat)
        expected = gs.array([.12])

        self.assertAllClose(result, expected)

    def test_rotation_vector_and_rotation_matrix(self):
        """
        This tests that the composition of
        rotation_vector_from_matrix
        and
        matrix_from_rotation_vector
        is the identity.
        """
        # TODO(nguigs): bring back a 1d representation of SO2
        point = gs.array([0.78])

        rot_mat = self.group.matrix_from_rotation_vector(point)
        result = self.group.rotation_vector_from_matrix(rot_mat)

        expected = point

        self.assertAllClose(result, expected)

    def test_rotation_vector_and_rotation_matrix_vectorization(self):
        rot_vecs = gs.array([
            [2.],
            [1.3],
            [0.8],
            [0.03]])

        rot_mats = self.group.matrix_from_rotation_vector(rot_vecs)
        result = self.group.rotation_vector_from_matrix(rot_mats)

        expected = self.group.regularize(rot_vecs)

        self.assertAllClose(result, expected)

    def test_compose(self):
        point_a = gs.array([.12])
        point_b = gs.array([-.15])
        result = self.group.compose(point_a, point_b)
        expected = self.group.regularize(gs.array([-.03]))
        self.assertAllClose(result, expected)

    def test_compose_and_inverse(self):
        angle = 0.986
        point = gs.array([angle])
        inv_point = self.group.inverse(point)
        result = self.group.compose(point, inv_point)
        expected = self.group.identity
        self.assertAllClose(result, expected)

        result = self.group.compose(inv_point, point)
        expected = self.group.identity
        self.assertAllClose(result, expected)

    def test_compose_vectorization(self):
        point_type = 'vector'
        self.group.default_point_type = point_type

        n_samples = self.n_samples
        n_points_a = self.group.random_uniform(n_samples=n_samples)
        n_points_b = self.group.random_uniform(n_samples=n_samples)
        one_point = self.group.random_uniform(n_samples=1)

        result = self.group.compose(one_point, n_points_a)
        self.assertAllClose(
            gs.shape(result), (n_samples, self.group.dim))

        result = self.group.compose(n_points_a, one_point)
        self.assertAllClose(
            gs.shape(result), (n_samples, self.group.dim))

        result = self.group.compose(n_points_a, n_points_b)
        self.assertAllClose(
            gs.shape(result), (n_samples, self.group.dim))

    def test_inverse_vectorization(self):
        n_samples = self.n_samples
        points = self.group.random_uniform(n_samples=n_samples)
        result = self.group.inverse(points)

        self.assertAllClose(
            gs.shape(result), (n_samples, self.group.dim))

    def test_group_exp(self):
        """
        The Riemannian exp and log are inverse functions of each other.
        This test is the inverse of test_log's.
        """
        rot_vec_base_point = gs.array([gs.pi / 5])
        rot_vec = gs.array([2 * gs.pi / 5])

        expected = gs.array([3 * gs.pi / 5])
        result = self.group.exp(base_point=rot_vec_base_point,
                                tangent_vec=rot_vec)
        self.assertAllClose(result, expected)

    def test_group_exp_vectorization(self):
        n_samples = self.n_samples

        one_tangent_vec = self.group.random_uniform(n_samples=1)
        one_base_point = self.group.random_uniform(n_samples=1)
        n_tangent_vec = self.group.random_uniform(n_samples=n_samples)
        n_base_point = self.group.random_uniform(n_samples=n_samples)

        # Test with the 1 base point, and n tangent vecs
        result = self.group.exp(n_tangent_vec, one_base_point)
        self.assertAllClose(
            gs.shape(result), (n_samples, self.group.dim))

        # Test with the several base point, and one tangent vec
        result = self.group.exp(one_tangent_vec, n_base_point)
        self.assertAllClose(
            gs.shape(result), (n_samples, self.group.dim))

        # Test with the same number n of base point and n tangent vec
        result = self.group.exp(n_tangent_vec, n_base_point)
        self.assertAllClose(
            gs.shape(result), (n_samples, self.group.dim))

    def test_group_log(self):
        """
        The Riemannian exp and log are inverse functions of each other.
        This test is the inverse of test_exp's.
        """
        rot_vec_base_point = gs.array([gs.pi / 5])
        rot_vec = gs.array([2 * gs.pi / 5])

        expected = gs.array([1 * gs.pi / 5])
        result = self.group.log(point=rot_vec, base_point=rot_vec_base_point)
        self.assertAllClose(result, expected)

    def test_group_log_vectorization(self):
        n_samples = self.n_samples

        one_point = self.group.random_uniform(n_samples=1)
        one_base_point = self.group.random_uniform(n_samples=1)
        n_point = self.group.random_uniform(n_samples=n_samples)
        n_base_point = self.group.random_uniform(n_samples=n_samples)

        # Test with the 1 base point, and several different points
        result = self.group.log(n_point, one_base_point)
        self.assertAllClose(
            gs.shape(result), (n_samples, self.group.dim))

        # Test with the several base point, and 1 point
        result = self.group.log(one_point, n_base_point)
        self.assertAllClose(
            gs.shape(result), (n_samples, self.group.dim))

        # Test with the same number n of base point and point
        result = self.group.log(n_point, n_base_point)
        self.assertAllClose(
            gs.shape(result), (n_samples, self.group.dim))

    def test_group_exp_then_log_from_identity(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        tangent_vec = gs.array([0.12])
        result = helper.group_exp_then_log_from_identity(
            group=self.group, tangent_vec=tangent_vec)
        expected = self.group.regularize(tangent_vec)
        self.assertAllClose(result, expected)

    def test_group_log_then_exp_from_identity(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        point = gs.array([0.12])
        result = helper.group_log_then_exp_from_identity(
            group=self.group, point=point)
        expected = self.group.regularize(point)
        self.assertAllClose(result, expected)

    def test_group_exp_then_log(self):
        """
        This tests that the composition of
        log and exp gives identity.

        """
        base_point = gs.array([0.12])
        tangent_vec = gs.array([.35])

        result = helper.group_exp_then_log(
            group=self.group,
            tangent_vec=tangent_vec,
            base_point=base_point)

        expected = self.group.regularize_tangent_vec(
            tangent_vec=tangent_vec,
            base_point=base_point)

        self.assertAllClose(result, expected, atol=1e-5)

    def test_group_log_then_exp(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        base_point = gs.array([0.12])
        point = gs.array([.35])

        result = helper.group_log_then_exp(
            group=self.group,
            point=point,
            base_point=base_point)

        expected = self.group.regularize(point)

        self.assertAllClose(result, expected, atol=1e-5)
