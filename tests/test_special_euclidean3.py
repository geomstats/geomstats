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
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.special_euclidean import SpecialEuclidean

# Tolerance for errors on predicted vectors, relative to the *norm*
# of the vector, as opposed to the standard behavior of gs.allclose
# where it is relative to each element of the array

RTOL = 1e-5


class TestSpecialEuclidean3Methods(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
        gs.random.seed(1234)

        group = SpecialEuclidean(n=3, point_type='vector')

        # Points

        # -- Rotation vectors with angles
        # 0, close to 0, closely lower than pi, pi,
        # between pi and 2pi, closely larger than 2pi, 2pi,
        # and closely larger than 2pi
        with_angle_0 = gs.zeros(6)
        with_angle_close_0 = (1e-10 * gs.array([1., -1., 1., 0., 0., 0.])
                              + gs.array([0., 0., 0., 1., 5., 2]))
        with_angle_close_pi_low = ((gs.pi - 1e-9) / gs.sqrt(2.)
                                   * gs.array([0., 1., -1., 0., 0., 0.])
                                   + gs.array([0., 0., 0., -100., 0., 2.]))
        with_angle_pi = (gs.pi / gs.sqrt(3.)
                         * gs.array([1., 1., -1., 0., 0., 0.])
                         + gs.array([0., 0., 0., -10.2, 0., 2.6]))
        with_angle_close_pi_high = ((gs.pi + 1e-9) / gs.sqrt(3.)
                                    * gs.array([-1., 1., -1., 0., 0., 0.])
                                    + gs.array([0., 0., 0., -100., 0., 2.]))
        with_angle_in_pi_2pi = ((gs.pi + 0.3) / gs.sqrt(5.)
                                * gs.array([-2., 1., 0., 0., 0., 0.])
                                + gs.array([0., 0., 0., -100., 0., 2.]))
        with_angle_close_2pi_low = ((2 * gs.pi - 1e-9) / gs.sqrt(6.)
                                    * gs.array([2., 1., -1., 0., 0., 0.])
                                    + gs.array([0., 0., 0., 8., 555., -2.]))
        with_angle_2pi = (2. * gs.pi / gs.sqrt(3.)
                          * gs.array([1., 1., -1., 0., 0., 0.])
                          + gs.array([0., 0., 0., 1., 8., -10.]))
        with_angle_close_2pi_high = ((2. * gs.pi + 1e-9) / gs.sqrt(2.)
                                     * gs.array([1., 0., -1., 0., 0., 0.])
                                     + gs.array([0., 0., 0., 1., 8., -10.]))

        point_1 = gs.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        point_2 = gs.array([0.5, 0., -0.3, 0.4, 5., 60.])

        translation_large = gs.array([0., 0., 0., 0.4, 0.5, 0.6])
        translation_small = gs.array([0., 0., 0., 0.5, 0.6, 0.7])
        rot_with_parallel_trans = gs.array([gs.pi / 3., 0., 0.,
                                           1., 0., 0.])

        elements_all = {
            'with_angle_0': with_angle_0,
            'with_angle_close_0': with_angle_close_0,
            'with_angle_close_pi_low': with_angle_close_pi_low,
            'with_angle_pi': with_angle_pi,
            'with_angle_close_pi_high': with_angle_close_pi_high,
            'with_angle_in_pi_2pi': with_angle_in_pi_2pi,
            'with_angle_close_2pi_low': with_angle_close_2pi_low,
            'with_angle_2pi': with_angle_2pi,
            'with_angle_close_2pi_high': with_angle_close_2pi_high,
            'translation_large': translation_large,
            'translation_small': translation_small,
            'point_1': point_1,
            'point_2': point_2,
            'rot_with_parallel_trans': rot_with_parallel_trans}
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

        # Metrics - only diagonals
        diag_mat_at_identity = gs.eye(6) * gs.array([2., 2., 2., 3., 3., 3.])

        left_diag_metric = InvariantMetric(
            group=group,
            inner_product_mat_at_identity=diag_mat_at_identity,
            left_or_right='left')
        right_diag_metric = InvariantMetric(
            group=group,
            inner_product_mat_at_identity=diag_mat_at_identity,
            left_or_right='right')

        # mat_at_identity = 7 * gs.eye(group.dim)

        # left_metric = InvariantMetric(
        #            group=group,
        #            inner_product_mat_at_identity=mat_at_identity,
        #            left_or_right='left')
        # right_metric = InvariantMetric(
        #            group=group,
        #            inner_product_mat_at_identity=mat_at_identity,
        #            left_or_right='right')

        metrics_all = {
            'left_canonical': group.left_canonical_metric,
            'right_canonical': group.right_canonical_metric,
            'left_diag': left_diag_metric,
            'right_diag': right_diag_metric}
        # FIXME:
        # 'left': left_metric,
        # 'right': right_metric}
        metrics = metrics_all
        if geomstats.tests.tf_backend():
            metrics = {'left_diag': left_diag_metric}

        self.group = group
        self.metrics_all = metrics_all
        self.metrics = metrics
        self.elements_all = elements_all
        self.elements = elements
        self.elements_matrices_all = elements_matrices_all
        self.elements_matrices = elements_matrices
        self.angles_close_to_pi_all = [
            'with_angle_close_pi_low',
            'with_angle_pi',
            'with_angle_close_pi_high']
        self.angles_close_to_pi = self.angles_close_to_pi_all
        if geomstats.tests.tf_backend():
            self.angles_close_to_pi = ['with_angle_close_pi_low']

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
        point = self.elements_all['with_angle_0']
        result = self.group.regularize(point)
        expected = point
        self.assertAllClose(result, expected)

        less_than_pi = ['with_angle_close_0',
                        'with_angle_close_pi_low']
        for angle_type in less_than_pi:
            point = self.elements_all[angle_type]
            result = self.group.regularize(point)
            expected = point
            self.assertAllClose(result, expected)

        if not geomstats.tests.tf_backend():
            # Note: by default, the rotation vector is inverted by
            # the function regularize when the angle of the rotation is pi.
            angle_type = 'with_angle_pi'
            point = self.elements_all[angle_type]
            result = self.group.regularize(point)

            expected = point

            self.assertAllClose(result, expected)

            angle_type = 'with_angle_close_pi_high'
            point = self.elements_all[angle_type]
            result = self.group.regularize(point)
            expected_rot = gs.concatenate(
                [point[:3] / gs.linalg.norm(point[:3]) * gs.pi,
                 gs.zeros(3)], axis=0)
            expected_trans = gs.concatenate(
                [gs.zeros(3), point[3:6]], axis=0)
            expected = expected_rot + expected_trans
            self.assertAllClose(result, expected)

            in_pi_2pi = ['with_angle_in_pi_2pi',
                         'with_angle_close_2pi_low']

            for angle_type in in_pi_2pi:
                point = self.elements_all[angle_type]
                angle = gs.linalg.norm(point[:3])
                new_angle = gs.pi - (angle - gs.pi)

                result = self.group.regularize(point)
                expected_rot = gs.concatenate(
                    [- new_angle * (point[:3] / angle),
                     gs.zeros(3)], axis=0)
                expected_trans = gs.concatenate(
                    [gs.zeros(3),
                     point[3:6]], axis=0)
                expected = expected_rot + expected_trans

                self.assertAllClose(result, expected)

            angle_type = 'with_angle_2pi'
            point = self.elements_all[angle_type]
            result = self.group.regularize(point)
            expected = gs.concatenate([gs.zeros(3), point[3:6]], axis=0)
            self.assertAllClose(result, expected)

            angle_type = 'with_angle_close_2pi_high'
            point = self.elements_all[angle_type]
            angle = gs.linalg.norm(point[:3])
            new_angle = angle - 2 * gs.pi

            result = self.group.regularize(point)

            expected_rot = gs.concatenate(
                [new_angle * point[:3] / angle,
                 gs.zeros(3)], axis=0)
            expected_trans = gs.concatenate(
                [gs.zeros(3), point[3:6]], axis=0)
            expected = expected_rot + expected_trans
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

    def test_left_jacobian_vectorization(self):
        n_samples = self.n_samples
        points = self.group.random_uniform(n_samples=n_samples)
        result = self.group.jacobian_translation(
            point=points, left_or_right='left')
        self.assertAllClose(
            gs.shape(result),
            (n_samples, *self.group.get_point_type_shape(),
             *self.group.get_point_type_shape()))

    def test_exp_from_identity_vectorization(self):
        n_samples = self.n_samples
        for metric in self.metrics.values():
            tangent_vecs = self.group.random_uniform(n_samples=n_samples)
            result = metric.exp_from_identity(tangent_vecs)

            self.assertAllClose(
                gs.shape(result),
                (n_samples, *self.group.get_point_type_shape()))

            if geomstats.tests.tf_backend():
                break

    def test_log_from_identity_vectorization(self):
        n_samples = self.n_samples
        for metric in self.metrics.values():
            points = self.group.random_uniform(n_samples=n_samples)
            result = metric.log_from_identity(points)

            self.assertAllClose(
                gs.shape(result),
                (n_samples, *self.group.get_point_type_shape()))

            if geomstats.tests.tf_backend():
                break

    def test_exp_vectorization(self):
        n_samples = self.n_samples

        for metric_type in self.metrics:
            metric = self.metrics[metric_type]

            one_tangent_vec = self.group.random_uniform(n_samples=1)
            one_base_point = self.group.random_uniform(n_samples=1)
            n_tangent_vec = self.group.random_uniform(n_samples=n_samples)
            n_base_point = self.group.random_uniform(n_samples=n_samples)

            # Test with the 1 base point, and n tangent vecs
            result = metric.exp(n_tangent_vec, one_base_point)
            self.assertAllClose(
                gs.shape(result),
                (n_samples, *self.group.get_point_type_shape()))

            if geomstats.tests.tf_backend():
                break

            # Test with the several base point, and one tangent vec
            result = metric.exp(one_tangent_vec, n_base_point)
            self.assertAllClose(
                gs.shape(result),
                (n_samples, *self.group.get_point_type_shape()))

            # Test with the same number n of base point and n tangent vec
            result = metric.exp(n_tangent_vec, n_base_point)
            self.assertAllClose(
                gs.shape(result),
                (n_samples, *self.group.get_point_type_shape()))

    def test_log_vectorization(self):
        n_samples = self.n_samples
        for metric_type in self.metrics:
            metric = self.metrics[metric_type]

            one_point = self.group.random_uniform(n_samples=1)
            one_base_point = self.group.random_uniform(n_samples=1)
            n_point = self.group.random_uniform(n_samples=n_samples)
            n_base_point = self.group.random_uniform(n_samples=n_samples)

            # Test with the 1 base point, and several different points
            result = metric.log(n_point, one_base_point)
            self.assertAllClose(
                gs.shape(result),
                (n_samples, *self.group.get_point_type_shape()))

            if geomstats.tests.tf_backend():
                break

            # Test with the several base point, and 1 point
            result = metric.log(one_point, n_base_point)
            self.assertAllClose(
                gs.shape(result),
                (n_samples, *self.group.get_point_type_shape()))

            # Test with the same number n of base point and point
            result = metric.log(n_point, n_base_point)
            self.assertAllClose(
                gs.shape(result),
                (n_samples, *self.group.get_point_type_shape()))

    @geomstats.tests.np_only
    def test_group_exp_from_identity_vectorization(self):
        n_samples = self.n_samples
        tangent_vecs = self.group.random_uniform(n_samples=n_samples)
        result = self.group.exp_from_identity(tangent_vecs)

        self.assertAllClose(
            gs.shape(result), (n_samples, *self.group.get_point_type_shape()))

    @geomstats.tests.np_only
    def test_group_log_from_identity_vectorization(self):
        n_samples = self.n_samples
        points = self.group.random_uniform(n_samples=n_samples)
        result = self.group.log_from_identity(points)

        self.assertAllClose(
            gs.shape(result),
            (n_samples, *self.group.get_point_type_shape()))

    @geomstats.tests.np_only
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

    @geomstats.tests.np_only
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

    @geomstats.tests.np_only
    def test_group_exp_from_identity(self):
        # Group exponential of a translation (no rotational part)
        # Expect the original translation
        tangent_vec = self.elements_all['translation_small']
        result = self.group.exp(
            base_point=self.group.identity, tangent_vec=tangent_vec)
        expected = tangent_vec
        self.assertAllClose(result, expected)

        if not geomstats.tests.tf_backend():
            # Group exponential of a transformation
            # where translation is parallel to rotation axis
            # Expect the original transformation
            tangent_vec = self.elements_all['rot_with_parallel_trans']
            result = self.group.exp(
                base_point=self.group.identity,
                tangent_vec=tangent_vec)
            expected = tangent_vec
            self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_group_log_from_identity(self):
        # Group logarithm of a translation (no rotational part)
        # Expect the original translation
        point = self.elements_all['translation_small']
        result = self.group.log(
            base_point=self.group.identity, point=point)
        expected = point
        self.assertAllClose(result, expected)

        if not geomstats.tests.tf_backend():
            # Group logarithm of a transformation
            # where translation is parallel to rotation axis
            # Expect the original transformation
            point = self.elements_all['rot_with_parallel_trans']
            result = self.group.log(
                base_point=self.group.identity, point=point)
            expected = point
            self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_group_log_then_exp_from_identity(self):
        """
        Test that the group exponential from the identity
        and the group logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        for element_type in self.elements:
            if element_type in self.angles_close_to_pi:
                continue
            point = self.elements[element_type]
            result = helper.group_log_then_exp_from_identity(
                group=self.group, point=point)
            expected = self.group.regularize(point)
            self.assertAllClose(result, expected, atol=1e-3)

            if geomstats.tests.tf_backend():
                break

    @geomstats.tests.np_only
    def test_group_log_then_exp_from_identity_with_angles_close_to_pi(self):
        """
        Test that the group exponential from the identity
        and the group logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        angle_types = self.angles_close_to_pi
        for element_type in angle_types:
            point = self.elements_all[element_type]
            result = helper.group_log_then_exp_from_identity(
                group=self.group, point=point)
            expected = self.group.regularize(point)

            inv_expected = gs.concatenate(
                [- expected[:3], expected[3:6]])

            self.assertTrue(
                gs.allclose(result, expected, atol=1e-4)
                or gs.allclose(result, inv_expected, atol=1e-4))

            if geomstats.tests.tf_backend():
                break

    @geomstats.tests.np_only
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

    @geomstats.tests.np_only
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

    @geomstats.tests.np_only
    def test_group_log_then_exp(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        for base_point in self.elements.values():
            for element_type in self.elements:
                if element_type in self.angles_close_to_pi:
                    continue
                point = self.elements[element_type]

                result = helper.group_log_then_exp(group=self.group,
                                                   point=point,
                                                   base_point=base_point)
                expected = self.group.regularize(point)
                self.assertAllClose(result, expected, rtol=1e-4, atol=1e-4)

                if geomstats.tests.tf_backend():
                    break

    @geomstats.tests.np_only
    def test_group_exp_then_log(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        for base_point_type in self.elements:
            base_point = self.elements[base_point_type]
            for element_type in self.elements:
                if element_type in self.angles_close_to_pi:
                    continue
                tangent_vec = self.elements[element_type]
                result = helper.group_exp_then_log(
                    group=self.group,
                    tangent_vec=tangent_vec,
                    base_point=base_point)
                metric = self.metrics_all['left_canonical']
                expected = self.group.regularize_tangent_vec(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    metric=metric)
                self.assertAllClose(result, expected, rtol=1e-4, atol=1e-4)

                if geomstats.tests.tf_backend():
                    break

    @geomstats.tests.np_only
    def test_exp_from_identity_left(self):
        # Riemannian left-invariant metric given by
        # the canonical inner product on the lie algebra
        # Expect the identity function
        # because we use the Riemannian left logarithm with canonical
        # inner product to parameterize the transformations
        metric = self.metrics_all['left_canonical']
        # General case
        tangent_rot_vec = gs.array([1., 1., 1.])  # NB: Regularized
        tangent_translation = gs.array([1., 0., -3.])
        tangent_vec = gs.concatenate(
            [tangent_rot_vec, tangent_translation],
            axis=0)
        result = metric.exp_from_identity(tangent_vec)
        expected = tangent_vec

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_log_from_identity_left(self):
        # Riemannian left-invariant metric given by
        # the canonical inner product on the lie algebra
        # Expect the identity function
        # because we use the Riemannian left logarithm with canonical
        # inner product to parameterize the transformations

        metric = self.metrics_all['left_canonical']
        # General case
        rot_vec = gs.array([0.1, 1, 0.9])  # NB: Regularized
        translation = gs.array([1., -19., -3.])
        transfo = gs.concatenate(
            [rot_vec, translation], axis=0)

        expected = transfo
        result = metric.log_from_identity(transfo)

        self.assertAllClose(result, expected)

        if not geomstats.tests.tf_backend():
            # Edge case: angle < epsilon, where angle = norm(rot_vec)
            rot_vec = gs.array([1e-8, 0., 1e-9])  # NB: Regularized
            translation = gs.array([10000., -5.9, -93])
            transfo = gs.concatenate(
                [rot_vec, translation], axis=0)

            expected = transfo
            result = metric.log_from_identity(transfo)

            self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_exp_then_log_from_identity_left(self):
        """
        Test that the Riemannian left exponential from the identity
        and the Riemannian left logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        # Canonical inner product on the lie algebra

        for metric in [self.metrics_all['left_canonical'],
                       self.metrics_all['left_diag']]:
            for angle_type in self.elements:
                if angle_type in self.angles_close_to_pi:
                    continue
                tangent_vec = self.elements[angle_type]
                result = helper.exp_then_log_from_identity(
                    metric=metric, tangent_vec=tangent_vec)
                expected = self.group.regularize_tangent_vec_at_identity(
                    tangent_vec, metric=metric)
                self.assertAllClose(result, expected)

                if geomstats.tests.tf_backend():
                    break

    @geomstats.tests.np_only
    def test_exp_then_log_from_identity_left_with_angles_close_to_pi(self):
        """
        Test that the Riemannian left exponential from the identity
        and the Riemannian left logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        angle_types = self.angles_close_to_pi
        # Canonical inner product on the lie algebra
        for metric in [self.metrics_all['left_canonical'],
                       self.metrics_all['left_diag']]:
            for angle_type in angle_types:
                tangent_vec = self.elements_all[angle_type]
                result = helper.exp_then_log_from_identity(
                    metric=metric, tangent_vec=tangent_vec)
                expected = self.group.regularize_tangent_vec_at_identity(
                    tangent_vec=tangent_vec, metric=metric)
                inv_expected = gs.concatenate(
                    [- expected[:3], expected[3:6]])

                self.assertTrue(
                    gs.allclose(result, expected)
                    or gs.allclose(result, inv_expected))

                if geomstats.tests.tf_backend():
                    break

    @geomstats.tests.np_only
    def test_exp_then_log_from_identity_right(self):
        """
        Test that the Riemannian right exponential from the identity
        and the Riemannian right logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        # Canonical inner product on the lie algebra
        for metric in [self.metrics_all['right_canonical'],
                       self.metrics_all['right_diag']]:
            for angle_type in self.elements:
                if angle_type in self.angles_close_to_pi:
                    continue
                tangent_vec = self.elements[angle_type]
                result = helper.exp_then_log_from_identity(
                    metric=metric, tangent_vec=tangent_vec)
                expected = self.group.regularize_tangent_vec_at_identity(
                    tangent_vec=tangent_vec, metric=metric)

                self.assertAllClose(result, expected, atol=1e-4)

                if geomstats.tests.tf_backend():
                    break

    @geomstats.tests.np_only
    def test_exp_then_log_from_identity_right_with_angles_close_to_pi(self):
        """
        Test that the Riemannian right exponential from the identity
        and the Riemannian right logarithm from the identity
        are inverse.
        Expect their composition to give the identity function.
        """
        if geomstats.tests.np_backend():
            atol = 1e-5
        else:
            atol = 5e-5
        angle_types = self.angles_close_to_pi
        # Canonical inner product on the lie algebra
        for metric in [self.metrics_all['right_canonical'],
                       self.metrics_all['right_diag']]:
            for angle_type in angle_types:
                tangent_vec = self.elements_all[angle_type]
                result = helper.exp_then_log_from_identity(
                    metric=metric, tangent_vec=tangent_vec)
                expected = self.group.regularize_tangent_vec_at_identity(
                    tangent_vec=tangent_vec, metric=metric)
                inv_expected = gs.concatenate(
                    [- expected[:3], expected[3:6]])
                self.assertTrue(
                    gs.allclose(result, expected, atol=atol)
                    or gs.allclose(result, inv_expected, atol=atol))

    @geomstats.tests.np_only
    def test_exp_left(self):
        # Reference point is a translation (no rotational part)
        # so that the jacobian of the left-translation of the Lie group
        # is the 6x6 identity matrix
        metric = self.metrics_all['left_canonical']
        rot_vec_base_point = gs.array([0., 0., 0.])
        translation_base_point = gs.array([4., -1., 10000.])
        transfo_base_point = gs.concatenate(
            [rot_vec_base_point, translation_base_point], axis=0)

        # Tangent vector is a translation (no infinitesimal rotational part)
        # Expect the sum of the translation
        # with the translation of the reference point
        rot_vec = gs.array([0., 0., 0.])
        translation = gs.array([1., 0., -3.])
        tangent_vec = gs.concatenate([rot_vec, translation], axis=0)

        result = metric.exp(base_point=transfo_base_point,
                            tangent_vec=tangent_vec)
        expected = gs.concatenate(
            [gs.array([0., 0., 0.]), gs.array([5., -1., 9997.])],
            axis=0)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_log_left(self):
        # Reference point is a translation (no rotational part)
        # so that the jacobian of the left-translation of the Lie group
        # is the 6x6 identity matrix
        metric = self.metrics_all['left_canonical']
        rot_vec_base_point = gs.array([0., 0., 0.])
        translation_base_point = gs.array([4., 0., 0.])
        transfo_base_point = gs.concatenate(
            [rot_vec_base_point, translation_base_point], axis=0)

        # Point is a translation (no rotational part)
        # Expect the difference of the translation
        # by the translation of the reference point
        rot_vec = gs.array([0., 0., 0.])
        translation = gs.array([-1., -1., -1.2])
        point = gs.concatenate(
            [rot_vec, translation], axis=0)

        expected = gs.concatenate(
            [gs.array([0., 0., 0.]), gs.array([-5., -1., -1.2])],
            axis=0)

        result = metric.log(base_point=transfo_base_point,
                            point=point)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_log_then_exp_left(self):
        """
        Test that the Riemannian left exponential and the
        Riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        for metric in [self.metrics_all['left_canonical'],
                       self.metrics_all['left_diag']]:
            for base_point_type in self.elements:
                base_point = self.elements[base_point_type]
                for element_type in self.elements:
                    if element_type in self.angles_close_to_pi:
                        continue
                    point = self.elements[element_type]
                    result = helper.log_then_exp(
                        metric=metric,
                        point=point,
                        base_point=base_point)

                    expected = self.group.regularize(point)

                    self.assertAllClose(result, expected, atol=1e-4)

                    if geomstats.tests.tf_backend():
                        break

    @geomstats.tests.np_only
    def test_log_then_exp_left_with_angles_close_to_pi(self):
        """
        Test that the Riemannian left exponential and the
        Riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        angle_types = self.angles_close_to_pi
        # Canonical inner product on the lie algebra
        for metric in [self.metrics_all['left_canonical'],
                       self.metrics_all['left_diag']]:
            for base_point in self.elements.values():
                for element_type in angle_types:
                    point = self.elements_all[element_type]
                    result = helper.log_then_exp(
                        metric=metric,
                        point=point,
                        base_point=base_point)

                    expected = self.group.regularize(point)

                    inv_expected = gs.concatenate(
                        [- expected[:3], expected[3:6]])

                    self.assertTrue(
                        gs.allclose(result, expected, atol=1e-6)
                        or gs.allclose(result, inv_expected, atol=1e-6))

                    if geomstats.tests.tf_backend():
                        break

    @geomstats.tests.np_only
    def test_exp_then_log_left(self):
        """
        Test that the Riemannian left exponential and the
        Riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        for metric in [self.metrics_all['left_canonical'],
                       self.metrics_all['left_diag']]:
            for base_point in self.elements.values():
                for element_type in self.elements:
                    if element_type in self.angles_close_to_pi:
                        continue
                    tangent_vec = self.elements[element_type]
                    result = helper.exp_then_log(
                        metric=metric,
                        tangent_vec=tangent_vec,
                        base_point=base_point)

                    expected = self.group.regularize_tangent_vec(
                        tangent_vec=tangent_vec,
                        base_point=base_point,
                        metric=metric)
                    norm = gs.linalg.norm(expected)
                    atol = RTOL
                    if norm != 0:
                        atol = RTOL * norm
                    self.assertAllClose(result, expected, atol=atol)

                    if geomstats.tests.tf_backend():
                        break

    @geomstats.tests.np_only
    def test_exp_then_log_left_with_angles_close_to_pi(self):
        """
        Test that the Riemannian left exponential and the
        Riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        angle_types = self.angles_close_to_pi
        # Canonical inner product on the lie algebra
        for metric in [self.metrics_all['left_canonical'],
                       self.metrics_all['left_diag']]:
            for base_point in self.elements.values():
                for element_type in angle_types:
                    tangent_vec = self.elements_all[element_type]
                    result = helper.exp_then_log(
                        metric=metric,
                        tangent_vec=tangent_vec,
                        base_point=base_point)

                    expected = self.group.regularize_tangent_vec(
                        tangent_vec=tangent_vec,
                        base_point=base_point,
                        metric=metric)

                    inv_expected = gs.concatenate(
                        [- expected[:3], expected[3:6]])

                    self.assertTrue(
                        gs.allclose(result, expected, atol=1e-3)
                        or gs.allclose(result, inv_expected, atol=1e-3))

                    if geomstats.tests.tf_backend():
                        break

    @geomstats.tests.np_only
    def test_log_then_exp_right(self):
        """
        Test that the Riemannian right exponential and the
        Riemannian right logarithm are inverse.
        Expect their composition to give the identity function.
        """
        for metric in [self.metrics_all['right_canonical'],
                       self.metrics_all['right_diag']]:
            for base_point in self.elements.values():
                for element_type in self.elements:
                    if element_type in self.angles_close_to_pi:
                        continue
                    point = self.elements[element_type]
                    result = helper.log_then_exp(
                        metric=metric,
                        point=point,
                        base_point=base_point)

                    expected = self.group.regularize(point)
                    norm = gs.linalg.norm(expected)
                    atol = RTOL
                    if norm != 0:
                        atol = RTOL * norm
                    self.assertAllClose(result, expected, atol=atol)

                    if geomstats.tests.tf_backend():
                        break

    @geomstats.tests.np_only
    def test_log_then_exp_right_with_angles_close_to_pi(self):
        """
        Test that the Riemannian right exponential and the
        Riemannian right logarithm are inverse.
        Expect their composition to give the identity function.
        """
        angle_types = self.angles_close_to_pi
        # Canonical inner product on the lie algebra
        for metric in [self.metrics_all['right_canonical'],
                       self.metrics_all['right_diag']]:
            for base_point in self.elements.values():
                for element_type in angle_types:
                    point = self.elements_all[element_type]
                    result = helper.log_then_exp(
                        metric=metric,
                        point=point,
                        base_point=base_point)

                    expected = self.group.regularize(point)

                    inv_expected = gs.concatenate(
                        [- expected[:3], expected[3:6]])

                    norm = gs.linalg.norm(expected)
                    atol = RTOL
                    if norm != 0:
                        atol = RTOL * norm

                    self.assertTrue(
                        gs.allclose(result, expected, atol=atol)
                        or gs.allclose(result, inv_expected, atol=atol))

                    if geomstats.tests.tf_backend():
                        break

    @geomstats.tests.np_only
    def test_exp_then_log_right(self):
        """
        Test that the Riemannian left exponential and the
        Riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # FIXME
        # for metric in [self.metrics_all['right_canonical'],
        #                self.metrics_all['right_diag']]:
        #     for base_point_type in self.elements:
        #         base_point = self.elements[base_point_type]
        #         for element_type in self.elements:
        #             if element_type in self.angles_close_to_pi:
        #                 continue
        #             tangent_vec = self.elements[element_type]
        #             result = helper.exp_then_log(
        #                                         metric=metric,
        #                                         tangent_vec=tangent_vec,
        #                                         base_point=base_point)

        #             expected = self.group.regularize_tangent_vec(
        #                                         tangent_vec=tangent_vec,
        #                                         base_point=base_point,
        #                                         metric=metric)

        #             if geomstats.tests.tf_backend():
        #                 break

    @geomstats.tests.np_only
    def test_exp_then_log_right_with_angles_close_to_pi(self):
        """
        Test that the Riemannian right exponential and the
        Riemannian right logarithm are inverse.
        Expect their composition to give the identity function.
        """
        # FIXME
        # angle_types = self.angles_close_to_pi
        # # Canonical inner product on the lie algebra
        # for metric in [self.metrics_all['right_canonical'],
        #                self.metrics_all['right_diag']]:
        #     for base_point in self.elements.values():
        #         for element_type in angle_types:
        #             tangent_vec = self.elements_all[element_type]
        #             result = helper.exp_then_log(
        #                                         metric=metric,
        #                                         tangent_vec=tangent_vec,
        #                                         base_point=base_point)

        #             expected = self.group.regularize_tangent_vec(
        #                                         tangent_vec=tangent_vec,
        #                                         base_point=base_point,
        #                                         metric=metric)

        #             inv_expected = gs.concatenate(
        #                 [- expected[:, :3], expected[:, 3:6]],
        #                 axis=1)
        #             norm = gs.linalg.norm(expected)
        #             atol = RTOL
        #             if norm != 0:
        #                 atol = RTOL * norm

        #             if geomstats.tests.tf_backend():
        #                 break

    def test_inner_product_at_identity_vectorization(self):
        n_samples = self.n_samples
        for metric in self.metrics.values():
            one_vector_a = self.group.random_uniform(n_samples=1)
            one_vector_b = self.group.random_uniform(n_samples=1)
            n_vector_a = self.group.random_uniform(n_samples=n_samples)
            n_vector_b = self.group.random_uniform(n_samples=n_samples)

            result = metric.inner_product(one_vector_a, n_vector_b)
            self.assertAllClose(gs.shape(result), (n_samples,))

            if geomstats.tests.tf_backend():
                break

            result = metric.inner_product(n_vector_a, one_vector_b)
            self.assertAllClose(gs.shape(result), (n_samples,))

            result = metric.inner_product(n_vector_a, n_vector_b)
            self.assertAllClose(gs.shape(result), (n_samples,))

    def test_inner_product_one_base_point_vectorization(self):
        n_samples = self.n_samples
        for metric in self.metrics.values():
            one_base_point = self.group.random_uniform(n_samples=1)

            one_vector_a = self.group.random_uniform(n_samples=1)
            one_vector_b = self.group.random_uniform(n_samples=1)
            n_vector_a = self.group.random_uniform(n_samples=n_samples)
            n_vector_b = self.group.random_uniform(n_samples=n_samples)

            result = metric.inner_product(one_vector_a, n_vector_b,
                                          one_base_point)
            self.assertAllClose(gs.shape(result), (n_samples,))

            if geomstats.tests.tf_backend():
                break

            result = metric.inner_product(n_vector_a, one_vector_b,
                                          one_base_point)
            self.assertAllClose(gs.shape(result), (n_samples,))

            result = metric.inner_product(n_vector_a, n_vector_b,
                                          one_base_point)
            self.assertAllClose(gs.shape(result), (n_samples,))

    def test_inner_product_n_base_point_vectorization(self):
        n_samples = self.n_samples
        for metric in self.metrics.values():
            n_base_point = self.group.random_uniform(n_samples=n_samples)

            one_vector_a = self.group.random_uniform(n_samples=1)
            one_vector_b = self.group.random_uniform(n_samples=1)
            n_vector_a = self.group.random_uniform(n_samples=n_samples)
            n_vector_b = self.group.random_uniform(n_samples=n_samples)

            result = metric.inner_product(one_vector_a, n_vector_b,
                                          n_base_point)
            self.assertAllClose(gs.shape(result), (n_samples,))

            if geomstats.tests.tf_backend():
                break

            result = metric.inner_product(n_vector_a, one_vector_b,
                                          n_base_point)
            self.assertAllClose(gs.shape(result), (n_samples,))

            result = metric.inner_product(n_vector_a, n_vector_b,
                                          n_base_point)
            self.assertAllClose(gs.shape(result), (n_samples,))

    @geomstats.tests.np_only
    def test_squared_dist_is_symmetric(self):
        for metric in self.metrics.values():
            for point_a in self.elements.values():
                for point_b in self.elements.values():
                    point_a = self.group.regularize(point_a)
                    point_b = self.group.regularize(point_b)

                    sq_dist_a_b = metric.squared_dist(point_a, point_b)
                    sq_dist_b_a = metric.squared_dist(point_b, point_a)

                    self.assertAllClose(sq_dist_a_b, sq_dist_b_a)

                    if geomstats.tests.tf_backend():
                        break

    @geomstats.tests.np_only
    def test_dist_is_symmetric(self):
        for metric in self.metrics.values():
            for point_a in self.elements.values():
                for point_b in self.elements.values():
                    point_a = self.group.regularize(point_a)
                    point_b = self.group.regularize(point_b)

                    dist_a_b = metric.dist(point_a, point_b)
                    dist_b_a = metric.dist(point_b, point_a)
                    self.assertAllClose(dist_a_b, dist_b_a)

                    if geomstats.tests.tf_backend():
                        break

    def test_squared_dist_vectorization(self):
        n_samples = self.n_samples
        for metric_type in self.metrics:
            metric = self.metrics[metric_type]
            point_id = self.group.identity

            one_point_1 = self.group.random_uniform(n_samples=1)
            one_point_2 = self.group.random_uniform(n_samples=1)
            one_point_1 = self.group.regularize(one_point_1)
            one_point_2 = self.group.regularize(one_point_2)

            n_point_1 = self.group.random_uniform(n_samples=n_samples)
            n_point_2 = self.group.random_uniform(n_samples=n_samples)
            n_point_1 = self.group.regularize(n_point_1)
            n_point_2 = self.group.regularize(n_point_2)

            # Identity and n points 2
            result = metric.squared_dist(point_id, n_point_2)
            self.assertAllClose(gs.shape(result), (n_samples,))

            if geomstats.tests.tf_backend():
                break

            # n points 1 and identity
            result = metric.squared_dist(n_point_1, point_id)
            self.assertAllClose(gs.shape(result), (n_samples,))

            # one point 1 and n points 2
            result = metric.squared_dist(one_point_1, n_point_2)
            self.assertAllClose(gs.shape(result), (n_samples,))

            # n points 1 and one point 2
            result = metric.squared_dist(n_point_1, one_point_2)
            self.assertAllClose(gs.shape(result), (n_samples,))

            # n points 1 and n points 2
            result = metric.squared_dist(n_point_1, n_point_2)
            self.assertAllClose(gs.shape(result), (n_samples,))

    def test_dist_vectorization(self):
        n_samples = self.n_samples
        for metric_type in self.metrics:
            metric = self.metrics[metric_type]
            point_id = self.group.identity

            one_point_1 = self.group.random_uniform(n_samples=1)
            one_point_2 = self.group.random_uniform(n_samples=1)
            one_point_1 = self.group.regularize(one_point_1)
            one_point_2 = self.group.regularize(one_point_2)

            n_point_1 = self.group.random_uniform(n_samples=n_samples)
            n_point_2 = self.group.random_uniform(n_samples=n_samples)
            n_point_1 = self.group.regularize(n_point_1)
            n_point_2 = self.group.regularize(n_point_2)

            # Identity and n points 2
            result = metric.dist(point_id, n_point_2)
            self.assertAllClose(gs.shape(result), (n_samples,))

            if geomstats.tests.tf_backend():
                break
            # n points 1 and identity
            result = metric.dist(n_point_1, point_id)
            self.assertAllClose(gs.shape(result), (n_samples,))

            # one point 1 and n points 2
            result = metric.dist(one_point_1, n_point_2)
            self.assertAllClose(gs.shape(result), (n_samples,))

            # n points 1 and one point 2
            result = metric.dist(n_point_1, one_point_2)
            self.assertAllClose(gs.shape(result), (n_samples,))

            # n points 1 and n points 2
            result = metric.dist(n_point_1, n_point_2)
            self.assertAllClose(gs.shape(result), (n_samples,))

    def test_geodesic_and_belongs(self):
        initial_point = self.group.random_uniform()
        initial_tangent_vec = gs.array([2., 0., -1., 0., 2., 3.])
        metric = self.metrics_all['left_canonical']
        geodesic = metric.geodesic(initial_point=initial_point,
                                   initial_tangent_vec=initial_tangent_vec)

        t = gs.linspace(start=0., stop=1., num=100)
        points = geodesic(t)
        result = gs.all(self.group.belongs(points))
        expected = True
        self.assertAllClose(result, expected)

    def test_geodesic_subsample(self):
        initial_point = gs.array([-1.1, 0., 0.99, 10., 2., 3.])
        initial_tangent_vec = gs.array([1., 0., 2., 1., 1., 1.])
        metric = self.metrics_all['left_canonical']
        geodesic = metric.geodesic(initial_point=initial_point,
                                   initial_tangent_vec=initial_tangent_vec)
        n_steps = 10
        t = gs.linspace(start=0., stop=1., num=n_steps + 1)
        points = geodesic(t)

        tangent_vec_step = initial_tangent_vec / n_steps
        for i in range(n_steps + 1):
            point_step = metric.exp(
                tangent_vec=i * tangent_vec_step,
                base_point=initial_point)
            result = point_step
            expected = points[i]
            self.assertAllClose(result, expected)
