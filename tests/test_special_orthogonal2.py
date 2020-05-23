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

        n_seq = [2]
        so = {n: SpecialOrthogonal(n=n, point_type='vector') for n in n_seq}

        # -- Set attributes
        self.n_seq = n_seq
        self.so = so

        self.n_samples = 4

    def test_projection(self):
        # Test 2D and nD cases
        for n in self.n_seq:
            group = self.so[n]
            rot_mat = gs.eye(n)
            delta = 1e-12 * gs.ones((n, n))
            rot_mat_plus_delta = rot_mat + delta
            result = group.projection(rot_mat_plus_delta)
            expected = rot_mat
            self.assertAllClose(result, expected)

    def test_projection_vectorization(self):
        for n in self.n_seq:
            group = self.so[n]
            n_samples = self.n_samples
            mats = gs.ones((n_samples, n, n))
            result = group.projection(mats)
            self.assertAllClose(gs.shape(result), (n_samples, n, n))

    def test_skew_matrix_from_vector(self):
        # Specific to 2D case
        n = 2
        group = self.so[n]
        rot_vec = gs.array([0.9])
        skew_matrix = group.skew_matrix_from_vector(rot_vec)
        result = gs.matmul(skew_matrix, skew_matrix)
        diag = gs.array([-0.81, -0.81])
        expected = algebra_utils.from_vector_to_diagonal_matrix(diag)
        self.assertAllClose(result, expected)

    def test_skew_matrix_and_vector(self):
        n = 2

        group = self.so[n]
        rot_vec = gs.array([0.8])

        skew_mat = group.skew_matrix_from_vector(rot_vec)
        result = group.vector_from_skew_matrix(skew_mat)
        expected = rot_vec

        self.assertAllClose(result, expected)

    def test_skew_matrix_from_vector_vectorization(self):
        n_samples = self.n_samples
        for n in self.n_seq:
            group = self.so[n]
            rot_vecs = group.random_uniform(n_samples=n_samples)
            result = group.skew_matrix_from_vector(rot_vecs)

            self.assertAllClose(gs.shape(result), (n_samples, n, n))

    def test_random_uniform_shape(self):
        group = self.so[2]
        result = group.random_uniform()
        self.assertAllClose(gs.shape(result), (group.dim,))

    def test_random_and_belongs(self):
        for n in self.n_seq:
            group = self.so[n]
            point = group.random_uniform()
            result = group.belongs(point)
            expected = True
            self.assertAllClose(result, expected)

    def test_random_and_belongs_vectorization(self):
        n_samples = self.n_samples
        for n in self.n_seq:
            group = self.so[n]
            points = group.random_uniform(n_samples=n_samples)
            result = group.belongs(points)
            expected = gs.array([True] * n_samples)
            self.assertAllClose(result, expected)

    def test_regularize(self):
        # Specific to 2D
        for n in self.n_seq:
            group = self.so[n]
            if n == 2:
                angle = 2 * gs.pi + 1
                result = group.regularize(gs.array([angle]))
                expected = gs.array([1.])
                self.assertAllClose(result, expected)

    def test_regularize_vectorization(self):
        for n in self.n_seq:
            group = self.so[n]

            n_samples = self.n_samples
            rot_vecs = group.random_uniform(n_samples=n_samples)
            result = group.regularize(rot_vecs)

            self.assertAllClose(gs.shape(result), (n_samples, group.dim))

    def test_matrix_from_rotation_vector(self):
        n = 2
        group = self.so[n]

        angle = gs.pi / 3
        expected = gs.array([[1. / 2, -gs.sqrt(3.) / 2],
                             [gs.sqrt(3.) / 2, 1. / 2]])
        result = group.matrix_from_rotation_vector(gs.array([angle]))
        self.assertAllClose(result, expected)

    def test_matrix_from_rotation_vector_vectorization(self):
        for n in self.n_seq:
            group = self.so[n]

            n_samples = self.n_samples
            rot_vecs = group.random_uniform(n_samples=n_samples)

            rot_mats = group.matrix_from_rotation_vector(rot_vecs)

            self.assertAllClose(
                gs.shape(rot_mats), (n_samples, group.n, group.n))

    def test_rotation_vector_from_matrix(self):
        n = 2
        group = self.so[n]

        angle = .12
        rot_mat = gs.array([[gs.cos(angle), -gs.sin(angle)],
                            [gs.sin(angle), gs.cos(angle)]])
        result = group.rotation_vector_from_matrix(rot_mat)
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
        for n in self.n_seq:
            group = self.so[n]
            if n == 2:
                # TODO(nguigs): bring back a 1d representation of SO2
                point = gs.array([0.78])

                rot_mat = group.matrix_from_rotation_vector(point)
                result = group.rotation_vector_from_matrix(rot_mat)

                expected = point

                self.assertAllClose(result, expected)

    def test_rotation_vector_and_rotation_matrix_vectorization(self):
        for n in self.n_seq:
            group = self.so[n]

            if n == 2:
                rot_vecs = gs.array([
                    [2.],
                    [1.3],
                    [0.8],
                    [0.03]])

            rot_mats = group.matrix_from_rotation_vector(rot_vecs)
            result = group.rotation_vector_from_matrix(rot_mats)

            expected = group.regularize(rot_vecs)

            self.assertAllClose(result, expected)

    def test_compose(self):
        for n in self.n_seq:
            group = self.so[n]
            if n == 2:
                point_a = gs.array([.12])
                point_b = gs.array([-.15])
                result = group.compose(point_a, point_b)
                expected = group.regularize(gs.array([-.03]))
                self.assertAllClose(result, expected)

    def test_compose_and_inverse(self):
        for n in self.n_seq:
            group = self.so[n]

            if n == 2:
                angle = 0.986
                point = gs.array([angle])
                inv_point = group.inverse(point)
                result = group.compose(point, inv_point)
                expected = group.identity
                self.assertAllClose(result, expected)

                result = group.compose(inv_point, point)
                expected = group.identity
                self.assertAllClose(result, expected)

    def test_compose_vectorization(self):
        point_type = 'vector'
        for n in self.n_seq:
            group = self.so[n]
            group.default_point_type = point_type

            n_samples = self.n_samples
            n_points_a = group.random_uniform(n_samples=n_samples)
            n_points_b = group.random_uniform(n_samples=n_samples)
            one_point = group.random_uniform(n_samples=1)

            result = group.compose(one_point, n_points_a)
            if point_type == 'vector':
                self.assertAllClose(
                    gs.shape(result), (n_samples, group.dim))
            if point_type == 'matrix':
                self.assertAllClose(
                    gs.shape(result), (n_samples, n, n))

            result = group.compose(n_points_a, one_point)
            if point_type == 'vector':
                self.assertAllClose(
                    gs.shape(result), (n_samples, group.dim))
            if point_type == 'matrix':
                self.assertAllClose(
                    gs.shape(result), (n_samples, n, n))

            result = group.compose(n_points_a, n_points_b)
            if point_type == 'vector':
                self.assertAllClose(
                    gs.shape(result), (n_samples, group.dim))
            if point_type == 'matrix':
                self.assertAllClose(
                    gs.shape(result), (n_samples, n, n))

    def test_inverse_vectorization(self):
        for n in self.n_seq:
            group = self.so[n]

            n_samples = self.n_samples
            points = group.random_uniform(n_samples=n_samples)
            result = group.inverse(points)

            if n == 2:
                self.assertAllClose(
                    gs.shape(result), (n_samples, group.dim))
            else:
                self.assertAllClose(
                    gs.shape(result), (n_samples, n, n))

    def test_group_exp(self):
        """
        The Riemannian exp and log are inverse functions of each other.
        This test is the inverse of test_log's.
        """
        n = 2
        group = self.so[n]

        rot_vec_base_point = gs.array([gs.pi / 5])
        rot_vec = gs.array([2 * gs.pi / 5])

        expected = gs.array([3 * gs.pi / 5])
        result = group.exp(base_point=rot_vec_base_point, tangent_vec=rot_vec)
        self.assertAllClose(result, expected)

    def test_group_exp_vectorization(self):
        n = 2
        group = self.so[n]

        n_samples = self.n_samples

        one_tangent_vec = group.random_uniform(n_samples=1)
        one_base_point = group.random_uniform(n_samples=1)
        n_tangent_vec = group.random_uniform(n_samples=n_samples)
        n_base_point = group.random_uniform(n_samples=n_samples)

        # Test with the 1 base point, and n tangent vecs
        result = group.exp(n_tangent_vec, one_base_point)
        self.assertAllClose(
            gs.shape(result), (n_samples, group.dim))

        # Test with the several base point, and one tangent vec
        result = group.exp(one_tangent_vec, n_base_point)
        self.assertAllClose(
            gs.shape(result), (n_samples, group.dim))

        # Test with the same number n of base point and n tangent vec
        result = group.exp(n_tangent_vec, n_base_point)
        self.assertAllClose(
            gs.shape(result), (n_samples, group.dim))

    def test_group_log(self):
        """
        The Riemannian exp and log are inverse functions of each other.
        This test is the inverse of test_exp's.
        """
        n = 2
        group = self.so[n]

        rot_vec_base_point = gs.array([gs.pi / 5])
        rot_vec = gs.array([2 * gs.pi / 5])

        expected = gs.array([1 * gs.pi / 5])
        result = group.log(point=rot_vec, base_point=rot_vec_base_point)
        self.assertAllClose(result, expected)

    def test_group_log_vectorization(self):
        n = 2
        group = self.so[n]

        n_samples = self.n_samples

        one_point = group.random_uniform(n_samples=1)
        one_base_point = group.random_uniform(n_samples=1)
        n_point = group.random_uniform(n_samples=n_samples)
        n_base_point = group.random_uniform(n_samples=n_samples)

        # Test with the 1 base point, and several different points
        result = group.log(n_point, one_base_point)
        self.assertAllClose(
            gs.shape(result), (n_samples, group.dim))

        # Test with the several base point, and 1 point
        result = group.log(one_point, n_base_point)
        self.assertAllClose(
            gs.shape(result), (n_samples, group.dim))

        # Test with the same number n of base point and point
        result = group.log(n_point, n_base_point)
        self.assertAllClose(
            gs.shape(result), (n_samples, group.dim))

    def test_group_exp_then_log_from_identity(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        n = 2
        group = self.so[n]

        tangent_vec = gs.array([0.12])
        result = helper.group_exp_then_log_from_identity(
            group=group, tangent_vec=tangent_vec)
        expected = group.regularize(tangent_vec)
        self.assertAllClose(result, expected)

    def test_group_log_then_exp_from_identity(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        n = 2
        group = self.so[n]

        point = gs.array([0.12])
        result = helper.group_log_then_exp_from_identity(
            group=group, point=point)
        expected = group.regularize(point)
        self.assertAllClose(result, expected)

    def test_group_exp_then_log(self):
        """
        This tests that the composition of
        log and exp gives identity.

        """
        n = 2
        group = self.so[n]

        base_point = gs.array([0.12])
        tangent_vec = gs.array([.35])

        result = helper.group_exp_then_log(
            group=group,
            tangent_vec=tangent_vec,
            base_point=base_point)

        expected = group.regularize_tangent_vec(
            tangent_vec=tangent_vec,
            base_point=base_point)

        self.assertAllClose(result, expected, atol=1e-5)

    def test_group_log_then_exp(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """

        n = 2
        group = self.so[n]

        base_point = gs.array([0.12])
        point = gs.array([.35])

        result = helper.group_log_then_exp(
            group=group,
            point=point,
            base_point=base_point)

        expected = group.regularize(point)

        self.assertAllClose(result, expected, atol=1e-5)
