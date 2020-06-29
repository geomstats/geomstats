"""Unit tests for special euclidean group SE(n).

Note: Only the *canonical* left- and right- invariant
metrics on SE(3) are tested here. Other invariant
metrics are tested with the tests of the invariant_metric
module.
"""

import warnings

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.special_euclidean import SpecialEuclidean

# Tolerance for errors on predicted vectors, relative to the *norm*
# of the vector, as opposed to the standard behavior of gs.allclose
# where it is relative to each element of the array

RTOL = 1e-5


class TestSpecialEuclidean2Methods(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
        gs.random.seed(1234)

        group = SpecialEuclidean(n=2, point_type='vector')

        point_1 = gs.array([0.1, 0.2, 0.3])
        point_2 = gs.array([0.5, 5., 60.])

        translation_large = gs.array([0., 5., 6.])
        translation_small = gs.array([0., 0.6, 0.7])

        elements_all = {
            'translation_large': translation_large,
            'translation_small': translation_small,
            'point_1': point_1,
            'point_2': point_2}
        elements = elements_all
        if geomstats.tests.tf_backend():
            # Tf is extremely slow
            elements = {
                'point_1': point_1,
                'point_2': point_2}

        elements_matrices_all = {
            key: group.matrix_from_vector(elements_all[key]) for key in
            elements_all}
        elements_matrices = elements_matrices_all

        self.group = group
        self.elements_all = elements_all
        self.elements = elements
        self.elements_matrices_all = elements_matrices_all
        self.elements_matrices = elements_matrices

        self.n_samples = 4

    def test_random_and_belongs(self):
        """Checks random_uniform and belongs

        Test that the random uniform method samples
        on the special euclidean group.
        """
        base_point = self.group.random_uniform()
        result = self.group.belongs(base_point)
        expected = True
        self.assertAllClose(result, expected)

    def test_random_and_belongs_vectorization(self):
        n_samples = self.n_samples
        points = self.group.random_uniform(n_samples=n_samples)
        result = self.group.belongs(points)
        expected = gs.array([True] * n_samples)
        self.assertAllClose(result, expected)

    def test_regularize(self):
        point = self.elements_all['point_1']
        result = self.group.regularize(point)
        expected = point
        self.assertAllClose(result, expected)

    def test_regularize_vectorization(self):
        n_samples = self.n_samples
        points = self.group.random_uniform(n_samples=n_samples)
        regularized_points = self.group.regularize(points)

        self.assertAllClose(
            gs.shape(regularized_points),
            (n_samples, *self.group.get_point_type_shape()))

    def test_compose(self):
        # Composition by identity, on the right
        # Expect the original transformation
        point = self.elements_all['point_1']
        result = self.group.compose(point,
                                    self.group.identity)
        expected = point
        self.assertAllClose(result, expected)

        if not geomstats.tests.tf_backend():
            # Composition by identity, on the left
            # Expect the original transformation
            result = self.group.compose(self.group.identity,
                                        point)
            expected = point
            self.assertAllClose(result, expected)

            # Composition of translations (no rotational part)
            # Expect the sum of the translations
            result = self.group.compose(self.elements_all['translation_small'],
                                        self.elements_all['translation_large'])
            expected = (self.elements_all['translation_small']
                        + self.elements_all['translation_large'])
            self.assertAllClose(result, expected)

    def test_compose_and_inverse(self):
        point = self.elements_all['point_1']
        inv_point = self.group.inverse(point)
        # Compose transformation by its inverse on the right
        # Expect the group identity
        result = self.group.compose(point, inv_point)
        expected = self.group.identity
        self.assertAllClose(result, expected)

        if not geomstats.tests.tf_backend():
            # Compose transformation by its inverse on the left
            # Expect the group identity
            result = self.group.compose(inv_point, point)
            expected = self.group.identity
            self.assertAllClose(result, expected)

    def test_compose_vectorization(self):
        n_samples = self.n_samples
        n_points_a = self.group.random_uniform(n_samples=n_samples)
        n_points_b = self.group.random_uniform(n_samples=n_samples)
        one_point = self.group.random_uniform(n_samples=1)

        result = self.group.compose(one_point,
                                    n_points_a)
        self.assertAllClose(
            gs.shape(result), (n_samples, *self.group.get_point_type_shape()))

        result = self.group.compose(n_points_a,
                                    one_point)

        if not geomstats.tests.tf_backend():
            self.assertAllClose(
                gs.shape(result),
                (n_samples, *self.group.get_point_type_shape()))

            result = self.group.compose(n_points_a,
                                        n_points_b)
            self.assertAllClose(
                gs.shape(result),
                (n_samples, *self.group.get_point_type_shape()))

    def test_inverse_vectorization(self):
        n_samples = self.n_samples
        points = self.group.random_uniform(n_samples=n_samples)
        result = self.group.inverse(points)
        self.assertAllClose(
            gs.shape(result), (n_samples, *self.group.get_point_type_shape()))

    def test_group_exp_from_identity_vectorization(self):
        n_samples = self.n_samples
        tangent_vecs = self.group.random_uniform(n_samples=n_samples)
        result = self.group.exp_from_identity(tangent_vecs)

        self.assertAllClose(
            gs.shape(result), (n_samples, *self.group.get_point_type_shape()))

    def test_group_log_from_identity_vectorization(self):
        n_samples = self.n_samples
        points = self.group.random_uniform(n_samples=n_samples)
        result = self.group.log_from_identity(points)

        self.assertAllClose(
            gs.shape(result),
            (n_samples, *self.group.get_point_type_shape()))

    def test_group_exp_vectorization(self):
        n_samples = self.n_samples
        # Test with the 1 base_point, and several different tangent_vecs
        tangent_vecs = self.group.random_uniform(n_samples=n_samples)
        base_point = self.group.random_uniform(n_samples=1)
        result = self.group.exp(tangent_vecs, base_point)

        self.assertAllClose(
            gs.shape(result), (n_samples, *self.group.get_point_type_shape()))

        if not geomstats.tests.tf_backend():
            # Test with the same number of base_points and tangent_vecs
            tangent_vecs = self.group.random_uniform(n_samples=n_samples)
            base_points = self.group.random_uniform(n_samples=n_samples)
            result = self.group.exp(tangent_vecs, base_points)

            self.assertAllClose(
                gs.shape(result),
                (n_samples, *self.group.get_point_type_shape()))

            # Test with the several base_points, and 1 tangent_vec
            tangent_vec = self.group.random_uniform(n_samples=1)
            base_points = self.group.random_uniform(n_samples=n_samples)
            result = self.group.exp(tangent_vec, base_points)

            self.assertAllClose(
                gs.shape(result),
                (n_samples, *self.group.get_point_type_shape()))

    def test_group_log_vectorization(self):
        n_samples = self.n_samples
        # Test with the 1 base point, and several different points
        points = self.group.random_uniform(n_samples=n_samples)
        base_point = self.group.random_uniform(n_samples=1)
        result = self.group.log(points, base_point)

        self.assertAllClose(
            gs.shape(result), (n_samples, *self.group.get_point_type_shape()))

        if not geomstats.tests.tf_backend():

            # Test with the same number of base points and points
            points = self.group.random_uniform(n_samples=n_samples)
            base_points = self.group.random_uniform(n_samples=n_samples)
            result = self.group.log(points, base_points)

            self.assertAllClose(
                gs.shape(result),
                (n_samples, *self.group.get_point_type_shape()))

            # Test with the several base points, and 1 point
            point = self.group.random_uniform(n_samples=1)
            base_points = self.group.random_uniform(n_samples=n_samples)
            result = self.group.log(point, base_points)

            self.assertAllClose(
                gs.shape(result),
                (n_samples, *self.group.get_point_type_shape()))

    def test_group_exp_from_identity(self):
        # Group exponential of a translation (no rotational part)
        # Expect the original translation
        tangent_vec = self.elements_all['translation_small']
        result = self.group.exp(
            base_point=self.group.identity, tangent_vec=tangent_vec)
        expected = tangent_vec
        self.assertAllClose(result, expected)

    def test_group_log_from_identity(self):
        # Group logarithm of a translation (no rotational part)
        # Expect the original translation
        point = self.elements_all['translation_small']
        result = self.group.log(
            base_point=self.group.identity, point=point)
        expected = point
        self.assertAllClose(result, expected)

    def test_group_log_then_exp_from_identity(self):
        """
        Test that the group exponential from the identity
        and the group logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        for element_type in self.elements:
            point = self.elements[element_type]
            result = helper.group_log_then_exp_from_identity(
                group=self.group, point=point)
            expected = self.group.regularize(point)
            self.assertAllClose(result, expected, atol=1e-3)

            if geomstats.tests.tf_backend():
                break

    def test_group_exp(self):
        # Reference point is a translation (no rotational part)
        # so that the jacobian of the left-translation of the Lie group
        # is the 6x6 identity matrix
        # Tangent vector is a translation (no infinitesimal rotational part)
        # Expect the sum of the translation
        # with the translation of the reference point
        result = self.group.exp(
            base_point=self.elements_all['translation_small'],
            tangent_vec=self.elements_all['translation_large'])
        expected = (self.elements_all['translation_small']
                    + self.elements_all['translation_large'])
        self.assertAllClose(result, expected)

    def test_group_log(self):
        # Reference point is a translation (no rotational part)
        # so that the jacobian of the left-translation of the Lie group
        # is the 6x6 identity matrix
        # Point is a translation (no rotational part)
        # Expect the difference of the translation
        # by the translation of the reference point
        result = self.group.log(
            base_point=self.elements_all['translation_small'],
            point=self.elements_all['translation_large'])
        expected = (self.elements_all['translation_large']
                    - self.elements_all['translation_small'])

        self.assertAllClose(result, expected)

    def test_group_log_then_exp(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        for base_point in self.elements.values():
            for element_type in self.elements:
                point = self.elements[element_type]

                result = helper.group_log_then_exp(group=self.group,
                                                   point=point,
                                                   base_point=base_point)
                expected = self.group.regularize(point)
                self.assertAllClose(result, expected, rtol=1e-4, atol=1e-4)

                if geomstats.tests.tf_backend():
                    break

    def test_group_exp_then_log(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        for base_point_type in self.elements:
            base_point = self.elements[base_point_type]
            for element_type in self.elements:
                tangent_vec = self.elements[element_type]
                result = helper.group_exp_then_log(
                    group=self.group,
                    tangent_vec=tangent_vec,
                    base_point=base_point)
                expected = self.group.regularize_tangent_vec(
                    tangent_vec=tangent_vec,
                    base_point=base_point)
                self.assertAllClose(result, expected, rtol=1e-4, atol=1e-4)

                if geomstats.tests.tf_backend():
                    break
