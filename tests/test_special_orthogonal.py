"""
Unit tests for special orthogonal group SO(n).
"""

import warnings

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


EPSILON = 1e-5
ATOL = 1e-5


# TODO(nina): Speed up tf tests


class TestSpecialOrthogonalMethods(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=UserWarning)

        gs.random.seed(1234)

        n_seq = [2, 3]
        so = {n: SpecialOrthogonal(n=n) for n in n_seq}

        # -- Rotation vectors with angles
        # 0, close to 0, closely lower than pi, pi,
        # between pi and 2pi, closely larger than 2pi, 2pi,
        # and closely larger than 2pi
        with_angle_0 = gs.zeros(3)
        with_angle_close_0 = 1e-10 * gs.array([1., -1., 1.])
        with_angle_close_pi_low = ((gs.pi - 1e-9) / gs.sqrt(2.)
                                   * gs.array([0., 1., -1.]))
        with_angle_pi = gs.pi / gs.sqrt(3.) * gs.array([1., 1., -1.])
        with_angle_close_pi_high = ((gs.pi + 1e-9) / gs.sqrt(3.)
                                    * gs.array([-1., 1., -1]))
        with_angle_in_pi_2pi = ((gs.pi + 0.3) / gs.sqrt(5.)
                                * gs.array([-2., 1., 0.]))
        with_angle_close_2pi_low = ((2. * gs.pi - 1e-9) / gs.sqrt(6.)
                                    * gs.array([2., 1., -1]))
        with_angle_2pi = 2. * gs.pi / gs.sqrt(3.) * gs.array([1., 1., -1.])
        with_angle_close_2pi_high = ((2. * gs.pi + 1e-9) / gs.sqrt(2.)
                                     * gs.array([1., 0., -1.]))

        elements_all = {
            3: {'with_angle_0': with_angle_0,
                'with_angle_close_0': with_angle_close_0,
                'with_angle_close_pi_low': with_angle_close_pi_low,
                'with_angle_pi': with_angle_pi,
                'with_angle_close_pi_high': with_angle_close_pi_high,
                'with_angle_in_pi_2pi': with_angle_in_pi_2pi,
                'with_angle_close_2pi_low': with_angle_close_2pi_low,
                'with_angle_2pi': with_angle_2pi,
                'with_angle_close_2pi_high': with_angle_close_2pi_high}
        }
        elements = elements_all
        if geomstats.tests.tf_backend():
            # Tf is extremely slow
            elements = {
                3: {
                    'with_angle_in_pi_2pi': with_angle_in_pi_2pi,
                    'with_angle_close_pi_low': with_angle_close_pi_low}}
        # -- Metrics - only diagonals for now
        canonical_metrics = {n: group.bi_invariant_metric
                             for n, group in so.items()}

        diag_mats = {n: 9 * gs.eye(group.dimension)
                     for n, group in so.items()}
        left_diag_metrics = {
            n: InvariantMetric(
                group=group,
                inner_product_mat_at_identity=diag_mat,
                left_or_right='left')
            for n, group, diag_mat in zip(n_seq,
                                          so.values(),
                                          diag_mats.values())
        }

        right_diag_metrics = {
            n: InvariantMetric(
                group=group,
                inner_product_mat_at_identity=diag_mat,
                left_or_right='right')
            for n, group, diag_mat in zip(n_seq,
                                          so.values(),
                                          diag_mats.values())
        }

        mats = {2: 4 * gs.eye(1),
                3: 87 * gs.eye(3)}

        left_metrics = {
            n: InvariantMetric(
                group=group,
                inner_product_mat_at_identity=mat,
                left_or_right='left')
            for n, group, mat in zip(n_seq, so.values(), mats.values())
        }

        right_metrics = {
            n: InvariantMetric(
                group=group,
                inner_product_mat_at_identity=mat,
                left_or_right='right')
            for n, group, mat in zip(n_seq, so.values(), mats.values())
        }
        all_metrics = zip(n_seq,
                          canonical_metrics.values(),
                          left_diag_metrics.values(),
                          right_diag_metrics.values(),
                          left_metrics.values(),
                          right_metrics.values())

        metrics_all = {
            n: {'canonical': canonical,
                'left_diag': left_diag,
                'right_diag': right_diag,
                'left': left,
                'right': right}
            for n, canonical, left_diag, right_diag, left, right in all_metrics
        }
        metrics = metrics_all
        if geomstats.tests.tf_backend():
            metrics = {
                n: {'right': InvariantMetric(
                    group=group,
                    inner_product_mat_at_identity=mat,
                    left_or_right='right')}
                for n, group, mat in zip(n_seq, so.values(), mats.values())
            }

        angles_close_to_pi_all = {
            3: ['with_angle_close_pi_low',
                'with_angle_pi',
                'with_angle_close_pi_high']
        }
        angles_close_to_pi = angles_close_to_pi_all
        if geomstats.tests.tf_backend():
            angles_close_to_pi = {
                3: ['with_angle_close_pi_low']
            }

        # -- Set attributes
        self.n_seq = n_seq
        self.so = so
        self.elements = elements
        self.elements_all = elements_all

        self.metrics = metrics
        self.metrics_all = metrics_all
        self.angles_close_to_pi = angles_close_to_pi
        self.angles_close_to_pi_all = angles_close_to_pi_all
        self.n_samples = 4

    @geomstats.tests.np_only
    def test_projection(self):
        # Test 3D and nD cases
        for n in self.n_seq:
            group = self.so[n]
            rot_mat = gs.eye(n)
            delta = 1e-12 * gs.ones((n, n))
            rot_mat_plus_delta = rot_mat + delta
            result = group.projection(rot_mat_plus_delta)
            expected = helper.to_matrix(rot_mat)
            self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_projection_vectorization(self):
        for n in self.n_seq:
            group = self.so[n]
            n_samples = self.n_samples
            mats = gs.ones((n_samples, n, n))
            result = group.projection(mats)
            self.assertAllClose(gs.shape(result), (n_samples, n, n))

    def test_skew_matrix_from_vector(self):
        # Specific to 3D case
        n = 3
        group = self.so[n]
        rot_vec = gs.array([0.9, -0.5, 1.1])
        skew_matrix = group.skew_matrix_from_vector(rot_vec)
        result = gs.dot(skew_matrix[0], rot_vec)
        expected = gs.zeros(n)

        self.assertAllClose(result, expected)

    def test_skew_matrix_and_vector(self):
        n = 3

        group = self.so[n]
        rot_vec = gs.array([0.8, 0.2, -0.1])

        skew_mat = group.skew_matrix_from_vector(rot_vec)
        result = group.vector_from_skew_matrix(skew_mat)
        expected = helper.to_vector(rot_vec)

        self.assertAllClose(result, expected)

    def test_skew_matrix_from_vector_vectorization(self):
        point_type = 'vector'
        n_samples = self.n_samples
        for n in self.n_seq:
            group = self.so[n]
            rot_vecs = group.random_uniform(
                n_samples=n_samples, point_type=point_type)
            result = group.skew_matrix_from_vector(rot_vecs)

            self.assertAllClose(gs.shape(result), (n_samples, n, n))

    @geomstats.tests.np_only
    def test_random_and_belongs(self):
        for point_type in ('vector', 'matrix'):
            for n in self.n_seq:
                group = self.so[n]
                point = group.random_uniform(point_type=point_type)
                result = group.belongs(point, point_type=point_type)
                expected = gs.array([[True]])
                self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_random_and_belongs_vectorization(self):
        n_samples = self.n_samples
        for n in self.n_seq:
            group = self.so[n]
            points = group.random_uniform(n_samples=n_samples)
            result = group.belongs(points)
            expected = gs.array([[True]] * n_samples)
            self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_regularize(self):
        # Specific to 3D
        for n in self.n_seq:
            group = self.so[n]

            if n == 3:
                point = self.elements_all[3]['with_angle_0']
                self.assertAllClose(gs.linalg.norm(point), 0.)
                result = group.regularize(point)
                expected = helper.to_vector(point)
                self.assertAllClose(result, expected)

                less_than_pi = ['with_angle_close_0',
                                'with_angle_close_pi_low']
                for angle_type in less_than_pi:
                    point = self.elements_all[3][angle_type]
                    result = group.regularize(point)
                    expected = helper.to_vector(point)
                    self.assertAllClose(result, expected)

                # Note: by default, the rotation vector is inverted by
                # the function regularize when the angle of the rotation is pi.
                angle_type = 'with_angle_pi'
                point = self.elements_all[3][angle_type]
                result = group.regularize(point)
                expected = helper.to_vector(point)
                self.assertAllClose(result, expected)

                angle_type = 'with_angle_close_pi_high'
                point = self.elements_all[3][angle_type]
                result = group.regularize(point)
                expected = helper.to_vector(
                    point / gs.linalg.norm(point) * gs.pi)
                self.assertAllClose(result, expected)

                in_pi_2pi = ['with_angle_in_pi_2pi',
                             'with_angle_close_2pi_low']

                for angle_type in in_pi_2pi:
                    point = self.elements_all[3][angle_type]
                    point_initial = point
                    angle = gs.linalg.norm(point)
                    new_angle = gs.pi - (angle - gs.pi)

                    point_initial = point
                    result = group.regularize(point)

                    expected = helper.to_vector(
                        - (new_angle / angle) * point_initial)
                    self.assertAllClose(result, expected)

                angle_type = 'with_angle_2pi'
                point = self.elements_all[3][angle_type]
                result = group.regularize(point)
                expected = gs.array([[0., 0., 0.]])
                self.assertAllClose(result, expected)

                angle_type = 'with_angle_close_2pi_high'
                point = self.elements_all[3][angle_type]
                angle = gs.linalg.norm(point)
                new_angle = angle - 2 * gs.pi

                result = group.regularize(point)
                expected = helper.to_vector(
                    new_angle * point / angle)
                self.assertAllClose(result, expected)

            else:
                angle = 0.345
                point = gs.array([[
                    [gs.cos(angle), -gs.sin(angle)],
                    [gs.sin(angle), gs.cos(angle)]]])
                result = group.regularize(point)
                expected = point
                self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_regularize_vectorization(self):
        for point_type in ('vector', 'matrix'):
            for n in self.n_seq:
                group = self.so[n]
                group.default_point_type = point_type

                n_samples = self.n_samples
                rot_vecs = group.random_uniform(n_samples=n_samples)
                result = group.regularize(rot_vecs)

                if point_type == 'vector':
                    self.assertAllClose(
                        gs.shape(result), (n_samples, group.dimension))
                if point_type == 'matrix':
                    self.assertAllClose(
                        gs.shape(result), (n_samples, n, n))

    def test_matrix_from_rotation_vector(self):
        n = 3
        group = self.so[n]

        rot_vec_0 = group.identity
        result = group.matrix_from_rotation_vector(rot_vec_0)
        expected = helper.to_matrix(gs.eye(3))
        self.assertAllClose(result, expected)

        rot_vec_1 = gs.array([gs.pi / 3., 0., 0.])
        result = group.matrix_from_rotation_vector(rot_vec_1)
        expected = gs.array([[
            [1., 0., 0.],
            [0., 0.5, -gs.sqrt(3.) / 2],
            [0., gs.sqrt(3.) / 2, 0.5]]])
        self.assertAllClose(result, expected)

        rot_vec_3 = 1e-11 * gs.array([12., 1., -81.])
        angle = gs.linalg.norm(rot_vec_3)
        skew_rot_vec_3 = 1e-11 * gs.array([[0., 81., 1.],
                                           [-81., 0., -12.],
                                           [-1., 12., 0.]])
        coef_1 = gs.sin(angle) / angle
        coef_2 = (1. - gs.cos(angle)) / (angle ** 2)
        expected = helper.to_matrix(
            gs.eye(3)
            + coef_1 * skew_rot_vec_3
            + coef_2 * gs.dot(skew_rot_vec_3,
                              skew_rot_vec_3))
        result = group.matrix_from_rotation_vector(rot_vec_3)
        self.assertAllClose(result, expected)

        rot_vec_6 = gs.array([.1, 1.3, -.5])
        angle = gs.linalg.norm(rot_vec_6)
        skew_rot_vec_6 = gs.array([[0., .5, 1.3],
                                   [-.5, 0., -.1],
                                   [-1.3, .1, 0.]])

        coef_1 = gs.sin(angle) / angle
        coef_2 = (1 - gs.cos(angle)) / (angle ** 2)
        result = group.matrix_from_rotation_vector(rot_vec_6)
        expected = helper.to_matrix(
            gs.eye(3)
            + coef_1 * skew_rot_vec_6
            + coef_2 * gs.dot(skew_rot_vec_6,
                              skew_rot_vec_6))
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_matrix_from_rotation_vector_vectorization(self):
        for n in self.n_seq:
            group = self.so[n]

            n_samples = self.n_samples
            rot_vecs = group.random_uniform(
                n_samples=n_samples, point_type='vector')

            rot_mats = group.matrix_from_rotation_vector(rot_vecs)

            self.assertAllClose(
                gs.shape(rot_mats), (n_samples, group.n, group.n))

    @geomstats.tests.np_only
    def test_rotation_vector_from_matrix(self):
        n = 3
        group = self.so[n]

        rot_mat = gs.array([[1., 0., 0.],
                            [0., gs.cos(.12), -gs.sin(.12)],
                            [0, gs.sin(.12), gs.cos(.12)]])
        result = group.rotation_vector_from_matrix(rot_mat)
        expected = .12 * gs.array([[1., 0., 0.]])

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
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

            if n == 3:
                for angle_type in self.elements[3]:
                    point = self.elements[3][angle_type]
                    if angle_type in self.angles_close_to_pi[3]:
                        continue

                    rot_mat = group.matrix_from_rotation_vector(point)
                    result = group.rotation_vector_from_matrix(rot_mat)

                    expected = group.regularize(point)

                    self.assertAllClose(result, expected)

            else:  # n == 2
                point = gs.array([[0.78]])

                rot_mat = group.matrix_from_rotation_vector(point)
                result = group.rotation_vector_from_matrix(rot_mat)

                expected = point

                self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_matrix_from_tait_bryan_angles_extrinsic_xyz(self):
        n = 3
        group = self.so[n]

        tait_bryan_angles = gs.array([0., 0., 0.])
        result = group.matrix_from_tait_bryan_angles_extrinsic_xyz(
            tait_bryan_angles)
        expected = helper.to_matrix(gs.eye(n))

        self.assertAllClose(result, expected)

        angle = gs.pi / 6.
        cos_angle = gs.cos(angle)
        sin_angle = gs.sin(angle)

        tait_bryan_angles = gs.array([angle, 0., 0.])
        result = group.matrix_from_tait_bryan_angles_extrinsic_xyz(
            tait_bryan_angles)
        expected = gs.array([[[cos_angle, - sin_angle, 0.],
                              [sin_angle, cos_angle, 0.],
                              [0., 0., 1.]]])

        self.assertAllClose(result, expected)

        tait_bryan_angles = gs.array([0., angle, 0.])
        result = group.matrix_from_tait_bryan_angles_extrinsic_xyz(
            tait_bryan_angles)
        expected = gs.array([[[cos_angle, 0., sin_angle],
                              [0., 1., 0.],
                              [- sin_angle, 0., cos_angle]]])

        self.assertAllClose(result, expected)

        tait_bryan_angles = gs.array([0., 0., angle])
        result = group.matrix_from_tait_bryan_angles_extrinsic_xyz(
            tait_bryan_angles)
        expected = gs.array([[[1., 0., 0.],
                              [0., cos_angle, - sin_angle],
                              [0., sin_angle, cos_angle]]])

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_matrix_from_tait_bryan_angles_extrinsic_zyx(self):
        n = 3
        group = self.so[n]

        tait_bryan_angles = gs.array([0., 0., 0.])
        result = group.matrix_from_tait_bryan_angles_extrinsic_zyx(
            tait_bryan_angles)
        expected = helper.to_matrix(gs.eye(n))

        self.assertAllClose(result, expected)

        angle = gs.pi / 6.
        cos_angle = gs.cos(angle)
        sin_angle = gs.sin(angle)

        tait_bryan_angles = gs.array([angle, 0., 0.])
        result = group.matrix_from_tait_bryan_angles_extrinsic_zyx(
            tait_bryan_angles)
        expected = gs.array([[[1., 0., 0.],
                              [0., cos_angle, - sin_angle],
                              [0., sin_angle, cos_angle]]])

        self.assertAllClose(result, expected)

        tait_bryan_angles = gs.array([0., angle, 0.])
        result = group.matrix_from_tait_bryan_angles_extrinsic_zyx(
            tait_bryan_angles)
        expected = gs.array([[[cos_angle, 0., sin_angle],
                              [0., 1., 0.],
                              [- sin_angle, 0., cos_angle]]])

        self.assertAllClose(result, expected)

        tait_bryan_angles = gs.array([0., 0., angle])
        result = group.matrix_from_tait_bryan_angles_extrinsic_zyx(
            tait_bryan_angles)
        expected = gs.array([[[cos_angle, - sin_angle, 0.],
                              [sin_angle, cos_angle, 0.],
                              [0., 0., 1.]]])

        self.assertAllClose(result, expected)

        angle_bis = gs.pi / 7.
        cos_angle_bis = gs.cos(angle_bis)
        sin_angle_bis = gs.sin(angle_bis)

        tait_bryan_angles = gs.array([angle, angle_bis, 0.])
        result = group.matrix_from_tait_bryan_angles_extrinsic_zyx(
            tait_bryan_angles)
        expected = gs.array([[[cos_angle_bis, 0., sin_angle_bis],
                              [sin_angle * sin_angle_bis,
                               cos_angle,
                               - sin_angle * cos_angle_bis],
                              [- cos_angle * sin_angle_bis,
                               sin_angle,
                               cos_angle * cos_angle_bis]]])

        self.assertAllClose(result, expected)

        tait_bryan_angles = gs.array([angle, 0., angle_bis])
        result = group.matrix_from_tait_bryan_angles_extrinsic_zyx(
            tait_bryan_angles)
        expected = gs.array([[[cos_angle_bis, - sin_angle_bis, 0.],
                              [cos_angle * sin_angle_bis,
                               cos_angle * cos_angle_bis,
                               - sin_angle],
                              [sin_angle * sin_angle_bis,
                               sin_angle * cos_angle_bis,
                               cos_angle]]])

        self.assertAllClose(result, expected)

        tait_bryan_angles = gs.array([0., angle, angle_bis])
        result = group.matrix_from_tait_bryan_angles_extrinsic_zyx(
            tait_bryan_angles)
        expected = gs.array([[[cos_angle * cos_angle_bis,
                               - cos_angle * sin_angle_bis,
                               sin_angle],
                              [sin_angle_bis, cos_angle_bis, 0.],
                              [- sin_angle * cos_angle_bis,
                               sin_angle * sin_angle_bis,
                               cos_angle]]])

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_matrix_from_tait_bryan_angles_intrinsic_xyz(self):
        """
        This tests that the rotation matrix computed from the
        Tait-Bryan angles [0, 0, 0] is the identiy as expected.
        """
        n = 3
        group = self.so[n]

        order = 'xyz'
        extrinsic_or_intrinsic = 'intrinsic'

        tait_bryan_angles = gs.array([0., 0., 0.])
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.to_ndarray(gs.eye(n), to_ndim=3)

        self.assertAllClose(result, expected)

        angle = gs.pi / 6.
        cos_angle = gs.cos(angle)
        sin_angle = gs.sin(angle)

        tait_bryan_angles = gs.array([angle, 0., 0.])
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[[cos_angle, - sin_angle, 0.],
                              [sin_angle, cos_angle, 0.],
                              [0., 0., 1.]]])

        self.assertAllClose(result, expected)

        tait_bryan_angles = gs.array([0., angle, 0.])
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[[cos_angle, 0., sin_angle],
                              [0., 1., 0.],
                              [- sin_angle, 0., cos_angle]]])

        self.assertAllClose(result, expected)

        tait_bryan_angles = gs.array([0., 0., angle])
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[[1., 0., 0.],
                              [0., cos_angle, - sin_angle],
                              [0., sin_angle, cos_angle]]])

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_matrix_from_tait_bryan_angles_intrinsic_zyx(self):
        """
        This tests that the matrix computed from the
        Tait-Bryan angles[0, 0, 0] is [1, 0., 0., 0.] as expected.
        """
        n = 3
        group = self.so[n]

        order = 'zyx'
        extrinsic_or_intrinsic = 'intrinsic'

        tait_bryan_angles = gs.array([0., 0., 0.])
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = helper.to_matrix(gs.eye(n))

        self.assertAllClose(result, expected)

        angle = gs.pi / 6.
        cos_angle = gs.cos(angle)
        sin_angle = gs.sin(angle)

        tait_bryan_angles = gs.array([angle, 0., 0.])
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[[1., 0., 0.],
                              [0., cos_angle, - sin_angle],
                              [0., sin_angle, cos_angle]]])

        self.assertAllClose(result, expected)

        tait_bryan_angles = gs.array([0., angle, 0.])
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[[cos_angle, 0., sin_angle],
                              [0., 1., 0.],
                              [- sin_angle, 0., cos_angle]]])

        self.assertAllClose(result, expected)

        tait_bryan_angles = gs.array([0., 0., angle])
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[[cos_angle, - sin_angle, 0.],
                              [sin_angle, cos_angle, 0.],
                              [0., 0., 1.]]])

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_tait_bryan_angles_from_matrix_extrinsic_xyz(self):
        n = 3
        group = self.so[n]
        extrinsic_or_intrinsic = 'extrinsic'
        order = 'xyz'

        matrix = gs.eye(n)
        result = group.tait_bryan_angles_from_matrix(
            matrix, extrinsic_or_intrinsic, order)
        expected = gs.array([[0., 0., 0.]])

        self.assertAllClose(result, expected)

        angle = gs.pi / 6.
        cos_angle = gs.cos(angle)
        sin_angle = gs.sin(angle)

        rot_mat = gs.array([[[1., 0., 0.],
                             [0., cos_angle, - sin_angle],
                             [0., sin_angle, cos_angle]]])
        result = group.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic_or_intrinsic, order)
        expected = gs.array([[0., 0., angle]])

        self.assertAllClose(result, expected)

        rot_mat = gs.array([[[cos_angle, 0., sin_angle],
                             [0., 1., 0.],
                             [- sin_angle, 0., cos_angle]]])
        result = group.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic_or_intrinsic, order)
        expected = gs.array([[0., angle, 0.]])

        self.assertAllClose(result, expected)

        rot_mat = gs.array([[[cos_angle, - sin_angle, 0.],
                             [sin_angle, cos_angle, 0.],
                             [0., 0., 1.]]])
        result = group.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic_or_intrinsic, order)
        expected = gs.array([[angle, 0., 0.]])

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_tait_bryan_angles_from_matrix_extrinsic_zyx(self):
        n = 3
        group = self.so[n]
        extrinsic_or_intrinsic = 'extrinsic'
        order = 'zyx'

        rot_mat = gs.eye(n)
        result = group.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic_or_intrinsic, order)
        expected = gs.array([[0., 0., 0.]])

        self.assertAllClose(result, expected)

        angle = gs.pi / 6.
        cos_angle = gs.cos(angle)
        sin_angle = gs.sin(angle)

        rot_mat = gs.array([[[1., 0., 0.],
                             [0., cos_angle, - sin_angle],
                             [0., sin_angle, cos_angle]]])
        result = group.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic_or_intrinsic, order)
        expected = gs.array([[angle, 0., 0.]])

        self.assertAllClose(result, expected)

        rot_mat = gs.array([[[cos_angle, 0., sin_angle],
                             [0., 1., 0.],
                             [- sin_angle, 0., cos_angle]]])
        result = group.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic_or_intrinsic, order)
        expected = gs.array([[0., angle, 0.]])

        self.assertAllClose(result, expected)

        rot_mat = gs.array([[[cos_angle, - sin_angle, 0.],
                             [sin_angle, cos_angle, 0.],
                             [0., 0., 1.]]])
        result = group.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic_or_intrinsic, order)
        expected = gs.array([[0., 0., angle]])

        self.assertAllClose(result, expected)

        angle_bis = gs.pi / 7.
        cos_angle_bis = gs.cos(angle_bis)
        sin_angle_bis = gs.sin(angle_bis)

        matrix = gs.array([[[cos_angle_bis, 0., sin_angle_bis],
                            [sin_angle * sin_angle_bis,
                             cos_angle,
                             - sin_angle * cos_angle_bis],
                            [- cos_angle * sin_angle_bis,
                             sin_angle,
                             cos_angle * cos_angle_bis]]])

        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[angle, angle_bis, 0.]])

        matrix = gs.array([[[cos_angle_bis, - sin_angle_bis, 0.],
                            [cos_angle * sin_angle_bis,
                             cos_angle * cos_angle_bis,
                             - sin_angle],
                            [sin_angle * sin_angle_bis,
                             sin_angle * cos_angle_bis,
                             cos_angle]]])

        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[angle, 0., angle_bis]])

        matrix = gs.array([[[cos_angle * cos_angle_bis,
                             - cos_angle * sin_angle_bis,
                             sin_angle],
                            [sin_angle_bis, cos_angle_bis, 0.],
                            [- sin_angle * cos_angle_bis,
                             sin_angle * sin_angle_bis,
                             cos_angle]]])

        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[0., angle, angle_bis]])

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_tait_bryan_angles_from_matrix_intrinsic_xyz(self):
        n = 3
        group = self.so[n]
        extrinsic_or_intrinsic = 'intrinsic'
        order = 'xyz'

        matrix = gs.eye(n)
        result = group.tait_bryan_angles_from_matrix(
            matrix, extrinsic_or_intrinsic, order)
        expected = gs.array([[0., 0., 0.]])

        self.assertAllClose(result, expected)

        angle = gs.pi / 6.
        cos_angle = gs.cos(angle)
        sin_angle = gs.sin(angle)

        rot_mat = gs.array([[[1., 0., 0.],
                             [0., cos_angle, - sin_angle],
                             [0., sin_angle, cos_angle]]])
        result = group.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic_or_intrinsic, order)
        expected = gs.array([[0., 0., angle]])

        self.assertAllClose(result, expected)

        rot_mat = gs.array([[[cos_angle, 0., sin_angle],
                             [0., 1., 0.],
                             [- sin_angle, 0., cos_angle]]])
        result = group.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic_or_intrinsic, order)
        expected = gs.array([[0., angle, 0.]])

        self.assertAllClose(result, expected)

        rot_mat = gs.array([[[cos_angle, - sin_angle, 0.],
                             [sin_angle, cos_angle, 0.],
                             [0., 0., 1.]]])
        result = group.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic_or_intrinsic, order)
        expected = gs.array([[angle, 0., 0.]])

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_tait_bryan_angles_from_matrix_intrinsic_zyx(self):
        n = 3
        group = self.so[n]
        extrinsic_or_intrinsic = 'intrinsic'
        order = 'zyx'

        rot_mat = gs.eye(n)
        result = group.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic_or_intrinsic, order)
        expected = gs.array([[0., 0., 0.]])

        self.assertAllClose(result, expected)

        angle = gs.pi / 6.
        cos_angle = gs.cos(angle)
        sin_angle = gs.sin(angle)

        rot_mat = gs.array([[[1., 0., 0.],
                             [0., cos_angle, - sin_angle],
                             [0., sin_angle, cos_angle]]])
        result = group.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic_or_intrinsic, order)
        expected = gs.array([[angle, 0., 0.]])

        self.assertAllClose(result, expected)

        rot_mat = gs.array([[[cos_angle, 0., sin_angle],
                             [0., 1., 0.],
                             [- sin_angle, 0., cos_angle]]])
        result = group.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic_or_intrinsic, order)
        expected = gs.array([[0., angle, 0.]])

        self.assertAllClose(result, expected)

        rot_mat = gs.array([[[cos_angle, - sin_angle, 0.],
                             [sin_angle, cos_angle, 0.],
                             [0., 0., 1.]]])
        result = group.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic_or_intrinsic, order)
        expected = gs.array([[0., 0., angle]])

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_matrix_and_tait_bryan_angles_extrinsic_xyz(self):
        """
        This tests that the composition of
        rotation_vector_from_tait_bryan_angles
        and
        tait_bryan_angles_from_rotation_vector
        is the identity.
        """
        n = 3
        group = self.so[n]

        order = 'xyz'
        extrinsic_or_intrinsic = 'extrinsic'

        point = gs.pi / (6. * gs.sqrt(3.)) * gs.array([1., 1., 1.])
        matrix = group.matrix_from_rotation_vector(point)

        tait_bryan_angles = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = matrix
        self.assertTrue(gs.allclose(result, expected),
                        ' for {} Tait-Bryan angles with order {}\n'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            extrinsic_or_intrinsic,
                            order,
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_matrix_and_tait_bryan_angles_extrinsic_zyx(self):
        """
        This tests that the composition of
        rotation_vector_from_tait_bryan_angles
        and
        tait_bryan_angles_from_rotation_vector
        is the identity.
        """
        n = 3
        group = self.so[n]

        order = 'zyx'
        extrinsic_or_intrinsic = 'extrinsic'

        angle = gs.pi / 7.
        cos_angle = gs.cos(angle)
        sin_angle = gs.sin(angle)

        rot_mat = gs.array([[[1., 0., 0.],
                             [0., cos_angle, - sin_angle],
                             [0., sin_angle, cos_angle]]])
        tait_bryan_angles = group.tait_bryan_angles_from_matrix(
            rot_mat,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = rot_mat
        self.assertTrue(gs.allclose(result, expected),
                        ' result = {};'
                        ' expected = {}.'.format(
                            result,
                            expected))

        rot_mat = gs.array([[[cos_angle, 0., sin_angle],
                             [0., 1., 0.],
                             [- sin_angle, 0., cos_angle]]])
        tait_bryan_angles = group.tait_bryan_angles_from_matrix(
            rot_mat,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = rot_mat

        self.assertTrue(gs.allclose(result, expected),
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            result,
                            expected))

        rot_mat = gs.array([[[cos_angle, - sin_angle, 0.],
                             [sin_angle, cos_angle, 0.],
                             [0., 0., 1.]]])
        tait_bryan_angles = group.tait_bryan_angles_from_matrix(
            rot_mat,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = rot_mat
        self.assertTrue(gs.allclose(result, expected),
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            result,
                            expected))

        angle_bis = gs.pi / 8.
        cos_angle_bis = gs.cos(angle_bis)
        sin_angle_bis = gs.sin(angle_bis)

        rot_mat = gs.array([[[cos_angle_bis, 0., sin_angle_bis],
                             [sin_angle * sin_angle_bis,
                              cos_angle,
                              - sin_angle * cos_angle_bis],
                             [- cos_angle * sin_angle_bis,
                              sin_angle,
                              cos_angle * cos_angle_bis]]])
        # This matrix corresponds to tait-bryan angles (angle, angle_bis, 0.)

        tait_bryan_angles = group.tait_bryan_angles_from_matrix(
            rot_mat,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = rot_mat
        self.assertTrue(gs.allclose(result, expected),
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            result,
                            expected))

        point = gs.pi / (6. * gs.sqrt(3.)) * gs.array([0., 2., 1.])
        rot_mat = group.matrix_from_rotation_vector(point)

        tait_bryan_angles = group.tait_bryan_angles_from_matrix(
            rot_mat,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = rot_mat
        self.assertTrue(gs.allclose(result, expected),
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_tait_bryan_angles_and_matrix_extrinsic_xyz(self):
        """
        This tests that the composition of
        rotation_vector_from_tait_bryan_angles
        and
        tait_bryan_angles_from_rotation_vector
        is the identity.
        """
        n = 3
        group = self.so[n]

        order = 'xyz'
        extrinsic_or_intrinsic = 'extrinsic'

        tait_bryan_angles = gs.array([0., 0., 0.])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        angle = gs.pi / 6.

        tait_bryan_angles = gs.array([angle, 0., 0.])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        tait_bryan_angles = gs.array([0., angle, 0.])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        tait_bryan_angles = gs.array([0., 0., angle])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        tait_bryan_angles = gs.array([0.1, 0.7, 0.3])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_tait_bryan_angles_and_matrix_extrinsic_zyx(self):
        """
        This tests that the composition of
        rotation_vector_from_tait_bryan_angles
        and
        tait_bryan_angles_from_rotation_vector
        is the identity.
        """
        n = 3
        group = self.so[n]

        order = 'zyx'
        extrinsic_or_intrinsic = 'extrinsic'

        tait_bryan_angles = gs.array([0., 0., 0.])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        angle = gs.pi / 6.

        tait_bryan_angles = gs.array([angle, 0., 0.])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        tait_bryan_angles = gs.array([0., angle, 0.])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        tait_bryan_angles = gs.array([0., 0., angle])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        tait_bryan_angles = gs.array([0.3, 0.3, 0.])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_matrix_and_tait_bryan_angles_intrinsic_xyz(self):
        """
        This tests that the composition of
        rotation_vector_from_tait_bryan_angles
        and
        tait_bryan_angles_from_rotation_vector
        is the identity.
        """
        n = 3
        group = self.so[n]

        order = 'xyz'
        extrinsic_or_intrinsic = 'intrinsic'

        angle = gs.pi / 6.
        cos_angle = gs.cos(angle)
        sin_angle = gs.sin(angle)

        matrix = gs.array([[[1., 0., 0.],
                            [0., cos_angle, - sin_angle],
                            [0., sin_angle, cos_angle]]])
        tait_bryan_angles = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        matrix = gs.array([[[cos_angle, 0., sin_angle],
                            [0., 1., 0.],
                            [- sin_angle, 0., cos_angle]]])
        tait_bryan_angles = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = matrix
        self.assertTrue(gs.allclose(result, expected),
                        ' for {} Tait-Bryan angles with order {}\n'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            extrinsic_or_intrinsic,
                            order,
                            result,
                            expected))

        expected = matrix
        self.assertTrue(gs.allclose(result, expected),
                        ' for {} Tait-Bryan angles with order {}\n'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            extrinsic_or_intrinsic,
                            order,
                            result,
                            expected))

        matrix = gs.array([[[cos_angle, - sin_angle, 0.],
                            [sin_angle, cos_angle, 0.],
                            [0., 0., 1.]]])
        tait_bryan_angles = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = matrix
        self.assertTrue(gs.allclose(result, expected),
                        ' for {} Tait-Bryan angles with order {}\n'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            extrinsic_or_intrinsic,
                            order,
                            result,
                            expected))

        point = gs.pi / (6. * gs.sqrt(3.)) * gs.array([1., 1., 1.])
        matrix = group.matrix_from_rotation_vector(point)

        tait_bryan_angles = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = matrix

        self.assertTrue(gs.allclose(result, expected),
                        ' for {} Tait-Bryan angles with order {}\n'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            extrinsic_or_intrinsic,
                            order,
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_matrix_and_tait_bryan_angles_intrinsic_zyx(self):
        """
        This tests that the composition of
        rotation_vector_from_tait_bryan_angles
        and
        tait_bryan_angles_from_rotation_vector
        is the identity.
        """
        n = 3
        group = self.so[n]

        order = 'zyx'
        extrinsic_or_intrinsic = 'intrinsic'

        point = gs.pi / (6. * gs.sqrt(3.)) * gs.array([1., 1., 1.])
        matrix = group.matrix_from_rotation_vector(point)

        tait_bryan_angles = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = matrix
        self.assertTrue(gs.allclose(result, expected),
                        ' for {} Tait-Bryan angles with order {}\n'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            extrinsic_or_intrinsic,
                            order,
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_tait_bryan_angles_and_matrix_intrinsic_xyz(self):
        """
        This tests that the composition of
        rotation_vector_from_tait_bryan_angles
        and
        tait_bryan_angles_from_rotation_vector
        is the identity.
        """
        n = 3
        group = self.so[n]

        order = 'xyz'
        extrinsic_or_intrinsic = 'intrinsic'

        tait_bryan_angles = gs.array([0., 0., 0.])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        angle = gs.pi / 6.

        tait_bryan_angles = gs.array([angle, 0., 0.])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        tait_bryan_angles = gs.array([0., angle, 0.])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        tait_bryan_angles = gs.array([0., 0., angle])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        tait_bryan_angles = gs.array([0.1, 0.7, 0.3])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_tait_bryan_angles_and_matrix_intrinsic_zyx(self):
        """
        This tests that the composition of
        rotation_vector_from_tait_bryan_angles
        and
        tait_bryan_angles_from_rotation_vector
        is the identity.
        """
        n = 3
        group = self.so[n]

        order = 'zyx'
        extrinsic_or_intrinsic = 'intrinsic'

        tait_bryan_angles = gs.array([0., 0., 0.])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        angle = gs.pi / 6.

        tait_bryan_angles = gs.array([angle, 0., 0.])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        tait_bryan_angles = gs.array([0., angle, 0.])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        tait_bryan_angles = gs.array([0., 0., angle])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        tait_bryan_angles = gs.array([0.1, 0.7, 0.3])
        matrix = group.matrix_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_matrix(
            matrix,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_quaternion_from_tait_bryan_angles_intrinsic_xyz(self):
        n = 3
        group = self.so[n]

        tait_bryan_angles = gs.array([0., 0., 0.])
        result = group.quaternion_from_tait_bryan_angles_intrinsic_xyz(
            tait_bryan_angles)
        expected = gs.array([[1., 0., 0., 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = {};'
                        ' expected = {}.'.format(
                            result,
                            expected))
        angle = gs.pi / 6.
        cos_half_angle = gs.cos(angle / 2.)
        sin_half_angle = gs.sin(angle / 2.)

        tait_bryan_angles = gs.array([angle, 0., 0.])
        result = group.quaternion_from_tait_bryan_angles_intrinsic_xyz(
            tait_bryan_angles)
        expected = gs.array([[cos_half_angle, 0., 0., sin_half_angle]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            result,
                            expected))

        tait_bryan_angles = gs.array([0., angle, 0.])
        result = group.quaternion_from_tait_bryan_angles_intrinsic_xyz(
            tait_bryan_angles)
        expected = gs.array([[cos_half_angle, 0., sin_half_angle, 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            result,
                            expected))

        tait_bryan_angles = gs.array([0., 0., angle])
        result = group.quaternion_from_tait_bryan_angles_intrinsic_xyz(
            tait_bryan_angles)
        expected = gs.array([[cos_half_angle, sin_half_angle, 0., 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_quaternion_from_tait_bryan_angles_intrinsic_zyx(self):
        n = 3
        group = self.so[n]
        extrinsic_or_intrinsic = 'intrinsic'
        order = 'zyx'

        tait_bryan_angles = gs.array([0., 0., 0.])
        result = group.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[1., 0., 0., 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = {};'
                        ' expected = {}.'.format(
                            result,
                            expected))
        angle = gs.pi / 6.
        cos_half_angle = gs.cos(angle / 2.)
        sin_half_angle = gs.sin(angle / 2.)

        tait_bryan_angles = gs.array([angle, 0., 0.])
        result = group.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[cos_half_angle, sin_half_angle, 0., 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            result,
                            expected))

        tait_bryan_angles = gs.array([0., angle, 0.])
        result = group.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[cos_half_angle, 0., sin_half_angle, 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            result,
                            expected))

        tait_bryan_angles = gs.array([0., 0., angle])
        result = group.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[cos_half_angle, 0., 0., sin_half_angle]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_tait_bryan_angles_from_quaternion_intrinsic_xyz(self):
        """
        This tests that the Tait-Bryan angles of the quaternion [1, 0, 0, 0],
        is [0, 0, 0] as expected.
        """
        n = 3
        group = self.so[n]

        order = 'xyz'
        extrinsic_or_intrinsic = 'intrinsic'

        quaternion = gs.array([1., 0., 0., 0.])
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[0., 0., 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = {};'
                        ' expected = {}.'.format(
                            result,
                            expected))

        angle = gs.pi / 6.
        cos_half_angle = gs.cos(angle / 2.)
        sin_half_angle = gs.sin(angle / 2.)

        quaternion = gs.array([cos_half_angle, sin_half_angle, 0., 0.])
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[0., 0., angle]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = {};'
                        ' expected = {}.'.format(
                            result,
                            expected))

        quaternion = gs.array([cos_half_angle, 0., sin_half_angle, 0.])
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[0., angle, 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = {};'
                        ' expected = {}.'.format(
                            result,
                            expected))

        quaternion = gs.array([cos_half_angle, 0., 0., sin_half_angle])
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[angle, 0., 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = {};'
                        ' expected = {}.'.format(
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_tait_bryan_angles_from_quaternion_intrinsic_zyx(self):
        """
        This tests that the Tait-Bryan angles of the quaternion [1, 0, 0, 0],
        is [0, 0, 0] as expected.
        """
        n = 3
        group = self.so[n]

        order = 'zyx'
        extrinsic_or_intrinsic = 'intrinsic'

        quaternion = gs.array([1., 0., 0., 0.])
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[0., 0., 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = {};'
                        ' expected = {}.'.format(
                            result,
                            expected))

        angle = gs.pi / 6.
        cos_half_angle = gs.cos(angle / 2.)
        sin_half_angle = gs.sin(angle / 2.)

        quaternion = gs.array([cos_half_angle, sin_half_angle, 0., 0.])
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[angle, 0., 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = {};'
                        ' expected = {}.'.format(
                            result,
                            expected))

        quaternion = gs.array([cos_half_angle, 0., sin_half_angle, 0.])
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[0., angle, 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = {};'
                        ' expected = {}.'.format(
                            result,
                            expected))

        quaternion = gs.array([cos_half_angle, 0., 0., sin_half_angle])
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[0., 0., angle]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = {};'
                        ' expected = {}.'.format(
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_quaternion_from_tait_bryan_angles_extrinsic_xyz(self):
        """
        This tests that the quaternion computed from the
        Tait-Bryan angles[0, 0, 0] is [1, 0., 0., 0.] as expected.
        """
        n = 3
        group = self.so[n]

        order = 'xyz'
        extrinsic_or_intrinsic = 'extrinsic'

        tait_bryan_angles = gs.array([0., 0., 0.])
        result = group.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[1., 0., 0., 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' for {} Tait-Bryan angles with order {}\n'
                        ' result = {};'
                        ' expected = {}.'.format(
                            extrinsic_or_intrinsic,
                            order,
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_quaternion_from_tait_bryan_angles_extrinsic_zyx(self):
        """
        This tests that the quaternion computed from the
        Tait-Bryan angles[0, 0, 0] is [1, 0., 0., 0.] as expected.
        """
        n = 3
        group = self.so[n]

        order = 'zyx'
        extrinsic_or_intrinsic = 'extrinsic'

        tait_bryan_angles = gs.array([0., 0., 0.])
        result = group.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[1., 0., 0., 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' for {} Tait-Bryan angles with order {}\n'
                        ' result = {};'
                        ' expected = {}.'.format(
                            extrinsic_or_intrinsic,
                            order,
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_tait_bryan_angles_from_quaternion_extrinsic_xyz(self):
        """
        This tests that the Tait-Bryan angles of the quaternion [1, 0, 0, 0],
        is [0, 0, 0] as expected.
        """
        n = 3
        group = self.so[n]

        order = 'xyz'
        extrinsic_or_intrinsic = 'extrinsic'

        quaternion = gs.array([1., 0., 0., 0.])
        result = group.tait_bryan_angles_from_quaternion(
            quaternion, extrinsic_or_intrinsic='intrinsic', order='zyx')
        expected = gs.array([[0., 0., 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' for {} Tait-Bryan angles with order {}\n'
                        ' result = {};'
                        ' expected = {}.'.format(
                            extrinsic_or_intrinsic,
                            order,
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_tait_bryan_angles_from_quaternion_extrinsic_zyx(self):
        """
        This tests that the Tait-Bryan angles of the quaternion [1, 0, 0, 0],
        is [0, 0, 0] as expected.
        """
        n = 3
        group = self.so[n]

        order = 'zyx'
        extrinsic_or_intrinsic = 'extrinsic'

        quaternion = gs.array([1., 0., 0., 0.])
        result = group.tait_bryan_angles_from_quaternion(
            quaternion, extrinsic_or_intrinsic='intrinsic', order='zyx')
        expected = gs.array([[0., 0., 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' for {} Tait-Bryan angles with order {}\n'
                        ' result = {};'
                        ' expected = {}.'.format(
                            extrinsic_or_intrinsic,
                            order,
                            result,
                            expected))

        angle = gs.pi / 6.
        cos_half_angle = gs.cos(angle / 2.)
        sin_half_angle = gs.sin(angle / 2.)

        quaternion = gs.array([cos_half_angle, sin_half_angle, 0., 0.])
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[angle, 0., 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = {};'
                        ' expected = {}.'.format(
                            result,
                            expected))

        quaternion = gs.array([cos_half_angle, 0., sin_half_angle, 0.])
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[0., angle, 0.]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = {};'
                        ' expected = {}.'.format(
                            result,
                            expected))

        quaternion = gs.array([cos_half_angle, 0., 0., sin_half_angle])
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        expected = gs.array([[0., 0., angle]])

        self.assertTrue(gs.allclose(result, expected),
                        ' result = {};'
                        ' expected = {}.'.format(
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_quaternion_and_tait_bryan_angles_extrinsic_xyz(self):
        """
        This tests that the composition of
        rotation_vector_from_tait_bryan_angles
        and
        tait_bryan_angles_from_rotation_vector
        is the identity.
        """
        n = 3
        group = self.so[n]

        order = 'xyz'
        extrinsic_or_intrinsic = 'extrinsic'

        for angle_type in self.elements[n]:
            point = self.elements[n][angle_type]
            if angle_type in self.angles_close_to_pi[n]:
                continue

            quaternion = group.quaternion_from_rotation_vector(point)

            tait_bryan_angles = group.tait_bryan_angles_from_quaternion(
                quaternion,
                extrinsic_or_intrinsic=extrinsic_or_intrinsic,
                order=order)
            result = group.quaternion_from_tait_bryan_angles(
                tait_bryan_angles,
                extrinsic_or_intrinsic=extrinsic_or_intrinsic,
                order=order)

            expected = quaternion

            self.assertTrue(gs.allclose(result, expected),
                            ' for {} Tait-Bryan angles with order {}\n'
                            'for point {}:\n'
                            ' result = {};'
                            ' expected = {}.'.format(
                                extrinsic_or_intrinsic,
                                order,
                                angle_type,
                                result,
                                expected))

        point = gs.pi / (6. * gs.sqrt(3.)) * gs.array([1., 1., 1.])
        quaternion = group.quaternion_from_rotation_vector(point)

        tait_bryan_angles = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = quaternion

        self.assertTrue(gs.allclose(result, expected),
                        ' for {} Tait-Bryan angles with order {}\n'
                        'for point {}:\n'
                        ' result = {};'
                        ' expected = {}.'.format(
                            extrinsic_or_intrinsic,
                            order,
                            angle_type,
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_quaternion_and_tait_bryan_angles_intrinsic_xyz(self):
        """
        This tests that the composition of
        rotation_vector_from_tait_bryan_angles
        and
        tait_bryan_angles_from_rotation_vector
        is the identity.
        """
        n = 3
        group = self.so[n]

        order = 'xyz'
        extrinsic_or_intrinsic = 'intrinsic'

        for angle_type in self.elements[n]:
            point = self.elements[n][angle_type]
            if angle_type in self.angles_close_to_pi[n]:
                continue

            quaternion = group.quaternion_from_rotation_vector(point)

            tait_bryan_angles = group.tait_bryan_angles_from_quaternion(
                quaternion,
                extrinsic_or_intrinsic=extrinsic_or_intrinsic,
                order=order)
            result = group.quaternion_from_tait_bryan_angles(
                tait_bryan_angles,
                extrinsic_or_intrinsic=extrinsic_or_intrinsic,
                order=order)

            expected = quaternion

            self.assertTrue(gs.allclose(result, expected),
                            ' for {} Tait-Bryan angles with order {}\n'
                            'for point {}:\n'
                            ' result = {};'
                            ' expected = {}.'.format(
                                extrinsic_or_intrinsic,
                                order,
                                angle_type,
                                result,
                                expected))

        point = gs.pi / (6 * gs.sqrt(3.)) * gs.array([1., 1., 1.])
        quaternion = group.quaternion_from_rotation_vector(point)

        tait_bryan_angles = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = quaternion

        self.assertTrue(gs.allclose(result, expected),
                        ' for {} Tait-Bryan angles with order {}\n'
                        'for point {}:\n'
                        ' result = {};'
                        ' expected = {}.'.format(
                            extrinsic_or_intrinsic,
                            order,
                            angle_type,
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_tait_bryan_angles_and_quaternion_intrinsic_xyz(self):
        n = 3
        group = self.so[n]

        order = 'xyz'
        extrinsic_or_intrinsic = 'intrinsic'

        tait_bryan_angles = gs.array([0., 0., 0.])
        quaternion = group.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        angle = gs.pi / 6.

        tait_bryan_angles = gs.array([angle, 0., 0.])
        quaternion = group.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        tait_bryan_angles = gs.array([0., angle, 0.])
        quaternion = group.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        tait_bryan_angles = gs.array([0., 0., angle])
        quaternion = group.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

        tait_bryan_angles = gs.array([0.1, 0.7, 0.3])
        quaternion = group.quaternion_from_tait_bryan_angles(
            tait_bryan_angles,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)
        result = group.tait_bryan_angles_from_quaternion(
            quaternion,
            extrinsic_or_intrinsic=extrinsic_or_intrinsic,
            order=order)

        expected = tait_bryan_angles

        self.assertTrue(gs.allclose(result, expected),
                        ' for tait-bryan angles = {}'
                        ' result = \n{};'
                        ' expected = \n{}.'.format(
                            tait_bryan_angles,
                            result,
                            expected))

    @geomstats.tests.np_only
    def test_rotation_vector_and_tait_bryan_angles_xyz(self):
        """
        This tests that the composition of
        rotation_vector_from_tait_bryan_angles
        and
        tait_bryan_angles_from_rotation_vector
        is the identity.
        """
        n = 3
        group = self.so[n]

        order = 'xyz'

        for extrinsic_or_intrinsic in ('extrinsic', 'intrinsic'):
            for angle_type in self.elements[n]:
                point = self.elements[n][angle_type]
                if angle_type in self.angles_close_to_pi[n]:
                    continue

                tait_bryan = group.tait_bryan_angles_from_rotation_vector(
                    point,
                    extrinsic_or_intrinsic=extrinsic_or_intrinsic,
                    order=order)
                result = group.rotation_vector_from_tait_bryan_angles(
                    tait_bryan,
                    extrinsic_or_intrinsic=extrinsic_or_intrinsic,
                    order=order)

                expected = group.regularize(point)

                self.assertTrue(gs.allclose(result, expected),
                                ' for {} Tait-Bryan angles with order {}\n'
                                'for point {}:\n'
                                ' result = {};'
                                ' expected = {}.'.format(
                                    extrinsic_or_intrinsic,
                                    order,
                                    angle_type,
                                    result,
                                    expected))

    @geomstats.tests.np_only
    def test_quaternion_and_tait_bryan_angles_extrinsic_zyx(self):
        """
        This tests that the composition of
        rotation_vector_from_tait_bryan_angles
        and
        tait_bryan_angles_from_rotation_vector
        is the identity.
        """
        n = 3
        group = self.so[n]

        order = 'zyx'
        extrinsic_or_intrinsic = 'extrinsic'

        for angle_type in self.elements[n]:
            point = self.elements[n][angle_type]
            if angle_type in self.angles_close_to_pi[n]:
                continue

            quaternion = group.quaternion_from_rotation_vector(point)

            tait_bryan_angles = group.tait_bryan_angles_from_quaternion(
                quaternion,
                extrinsic_or_intrinsic=extrinsic_or_intrinsic,
                order=order)
            result = group.quaternion_from_tait_bryan_angles(
                tait_bryan_angles,
                extrinsic_or_intrinsic=extrinsic_or_intrinsic,
                order=order)

            expected = quaternion

            self.assertTrue(gs.allclose(result, expected),
                            ' for {} Tait-Bryan angles with order {}\n'
                            'for point {}:\n'
                            ' result = {};'
                            ' expected = {}.'.format(
                                extrinsic_or_intrinsic,
                                order,
                                angle_type,
                                result,
                                expected))

    @geomstats.tests.np_only
    def test_quaternion_and_tait_bryan_angles_intrinsic_zyx(self):
        """
        This tests that the composition of
        rotation_vector_from_tait_bryan_angles
        and
        tait_bryan_angles_from_rotation_vector
        is the identity.
        """
        n = 3
        group = self.so[n]

        order = 'zyx'
        extrinsic_or_intrinsic = 'intrinsic'

        for angle_type in self.elements[n]:
            point = self.elements[n][angle_type]
            if angle_type in self.angles_close_to_pi[n]:
                continue

            quaternion = group.quaternion_from_rotation_vector(point)

            tait_bryan_angles = group.tait_bryan_angles_from_quaternion(
                quaternion,
                extrinsic_or_intrinsic=extrinsic_or_intrinsic,
                order=order)
            result = group.quaternion_from_tait_bryan_angles(
                tait_bryan_angles,
                extrinsic_or_intrinsic=extrinsic_or_intrinsic,
                order=order)

            expected = quaternion

            self.assertTrue(gs.allclose(result, expected),
                            ' for {} Tait-Bryan angles with order {}\n'
                            'for point {}:\n'
                            ' result = {};'
                            ' expected = {}.'.format(
                                extrinsic_or_intrinsic,
                                order,
                                angle_type,
                                result,
                                expected))

    @geomstats.tests.np_only
    def test_rotation_vector_and_rotation_matrix_vectorization(self):
        for n in self.n_seq:
            group = self.so[n]

            if n == 3:
                rot_vecs = gs.array([
                    [0.3, 0.2, 0.2],
                    [0., -0.4, 0.8],
                    [1.2, 0., 0.],
                    [1.1, 1.1, 0.]])
            if n == 2:
                rot_vecs = gs.array([
                    [2.],
                    [1.3],
                    [0.8],
                    [0.03]])

            rot_mats = group.matrix_from_rotation_vector(rot_vecs)
            result = group.rotation_vector_from_matrix(rot_mats)

            expected = group.regularize(
                rot_vecs, point_type='vector')

            self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_rotation_vector_and_rotation_matrix_with_angles_close_to_pi(self):
        """
        This tests that the composition of
        rotation_vector_from_matrix
        and
        matrix_from_rotation_vector
        is the identity.
        """
        n = 3
        group = self.so[n]

        angle_types = self.angles_close_to_pi[3]
        for angle_type in angle_types:
            point = self.elements[3][angle_type]

            rot_mat = group.matrix_from_rotation_vector(point)
            result = group.rotation_vector_from_matrix(rot_mat)

            expected = group.regularize(point)
            inv_expected = - expected

            self.assertTrue(
                gs.allclose(result, expected)
                or gs.allclose(result, inv_expected))

    @geomstats.tests.np_only
    def test_quaternion_and_rotation_vector(self):
        for n in self.n_seq:
            group = self.so[n]
            if n == 3:
                for angle_type in self.elements[3]:
                    point = self.elements[3][angle_type]
                    if angle_type in self.angles_close_to_pi[3]:
                        continue

                    quaternion = group.quaternion_from_rotation_vector(point)
                    result = group.rotation_vector_from_quaternion(quaternion)

                    expected = group.regularize(point)

                    self.assertAllClose(result, expected)

            else:
                point = group.random_uniform()
                self.assertRaises(
                    AssertionError,
                    lambda: group.quaternion_from_rotation_vector(point))
                fake_quaternion = gs.random.rand(1, n + 1)
                self.assertRaises(
                    AssertionError,
                    lambda: group.rotation_vector_from_quaternion(
                        fake_quaternion))

    def test_quaternion_and_rotation_vector_with_angles_close_to_pi(self):
        n = 3
        group = self.so[n]

        angle_types = self.angles_close_to_pi[3]
        for angle_type in angle_types:
            point = self.elements[3][angle_type]

            quaternion = group.quaternion_from_rotation_vector(point)
            result = group.rotation_vector_from_quaternion(quaternion)

            expected = group.regularize(point)
            inv_expected = - expected

            with self.session():
                self.assertTrue(
                    gs.eval(gs.allclose(result, expected))
                    or gs.eval(gs.allclose(result, inv_expected)))

    def test_quaternion_and_rotation_vector_vectorization(self):
        n = 3
        group = self.so[n]

        rot_vecs = gs.array([
            [1.2, 0., 0.9],
            [0.4, -0.5, 0.2],
            [0., 0., 1.9],
            [0.4, -0.12, 0.222]])
        quaternions = group.quaternion_from_rotation_vector(rot_vecs)
        result = group.rotation_vector_from_quaternion(quaternions)

        expected = group.regularize(rot_vecs)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_quaternion_and_matrix(self):
        for n in self.n_seq:
            group = self.so[n]

            if n == 3:
                for angle_type in self.elements[3]:
                    point = self.elements[3][angle_type]
                    if angle_type in self.angles_close_to_pi[3]:
                        continue

                    matrix = group.matrix_from_rotation_vector(point)

                    quaternion = group.quaternion_from_matrix(matrix)
                    result = group.matrix_from_quaternion(quaternion)

                    expected = matrix

                    self.assertAllClose(result, expected)

                angle = gs.pi / 9.
                cos_angle = gs.cos(angle)
                sin_angle = gs.sin(angle)

                angle_bis = gs.pi / 7.
                cos_angle_bis = gs.cos(angle_bis)
                sin_angle_bis = gs.sin(angle_bis)

                rot_mat = gs.array([[[cos_angle_bis, 0., sin_angle_bis],
                                     [sin_angle * sin_angle_bis,
                                      cos_angle,
                                      - sin_angle * cos_angle_bis],
                                     [- cos_angle * sin_angle_bis,
                                      sin_angle,
                                      cos_angle * cos_angle_bis]]])

                quaternion = group.quaternion_from_matrix(
                    rot_mat)
                result = group.matrix_from_quaternion(
                    quaternion)

                expected = rot_mat
                self.assertAllClose(result, expected)

                point = gs.pi / (6 * gs.sqrt(3.)) * gs.array([0., 2., 1.])
                rot_mat = group.matrix_from_rotation_vector(point)

                quaternion = group.quaternion_from_matrix(
                    rot_mat)
                result = group.matrix_from_quaternion(
                    quaternion)

                expected = rot_mat
                self.assertAllClose(result, expected)

            else:
                rot_vec = group.random_uniform(point_type='vector')

                rot_mat = group.matrix_from_rotation_vector(rot_vec)
                self.assertRaises(
                    AssertionError,
                    lambda: group.quaternion_from_matrix(rot_mat))
                fake_quaternion = gs.random.rand(1, n + 1)
                self.assertRaises(
                    AssertionError,
                    lambda: group.matrix_from_quaternion(
                        fake_quaternion))

    @geomstats.tests.np_only
    def test_quaternion_and_matrix_with_angles_close_to_pi(self):
        n = 3
        group = self.so[n]

        angle_types = self.angles_close_to_pi[3]
        for angle_type in angle_types:
            point = self.elements[3][angle_type]
            matrix = group.matrix_from_rotation_vector(point)

            quaternion = group.quaternion_from_matrix(matrix)
            result = group.matrix_from_quaternion(quaternion)

            expected = matrix
            inv_expected = gs.linalg.inv(matrix)

            self.assertTrue(
                gs.eval(gs.allclose(result, expected))
                or gs.eval(gs.allclose(result, inv_expected)))

    @geomstats.tests.np_only
    def test_quaternion_and_rotation_vector_and_matrix_vectorization(self):
        n = 3
        group = self.so[n]

        rot_vecs = gs.array([
            [0.2, 0., -0.3],
            [0.11, 0.11, 0.11],
            [-0.4, 0.2, 0.2],
            [0.66, -0.99, 0.]])
        rot_mats = group.matrix_from_rotation_vector(rot_vecs)

        quaternions = group.quaternion_from_matrix(rot_mats)
        result = group.matrix_from_quaternion(quaternions)

        expected = rot_mats
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_compose(self):
        for n in self.n_seq:
            group = self.so[n]
            if n == 3:
                for element_type in self.elements[3]:
                    point = self.elements[3][element_type]
                    # Composition by identity, on the right
                    # Expect the original transformation
                    result = group.compose(point, group.identity)
                    expected = group.regularize(point)
                    if element_type not in self.angles_close_to_pi[3]:
                        self.assertAllClose(result, expected)

                    else:
                        inv_expected = - expected
                        self.assertTrue(
                            gs.eval(gs.allclose(result, expected))
                            or gs.eval(gs.allclose(result, inv_expected)))

                    # Composition by identity, on the left
                    # Expect the original transformation
                    result = group.compose(group.identity, point)
                    expected = group.regularize(point)

                    if element_type not in self.angles_close_to_pi[3]:
                        self.assertAllClose(result, expected)
                    else:
                        inv_expected = - expected
                        self.assertTrue(
                            gs.eval(gs.allclose(result, expected))
                            or gs.eval(gs.allclose(result, inv_expected)))

            else:
                angle = 0.986
                point = gs.array([
                    [gs.cos(angle), -gs.sin(angle)],
                    [gs.sin(angle), gs.cos(angle)]])

                result = group.compose(point, group.identity)
                expected = group.regularize(point)
                self.assertAllClose(result, expected)

                result = group.compose(group.identity, point)
                expected = group.regularize(point)
                self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_compose_and_inverse(self):
        for n in self.n_seq:
            group = self.so[n]

            if n == 3:
                for point in self.elements[3].values():
                    inv_point = group.inverse(point)
                    # Compose transformation by its inverse on the right
                    # Expect the group identity
                    result = group.compose(point, inv_point)
                    expected = helper.to_vector(group.identity)
                    self.assertAllClose(result, expected)

                    # Compose transformation by its inverse on the left
                    # Expect the group identity
                    result = group.compose(inv_point, point)
                    expected = helper.to_vector(group.identity)
                    self.assertAllClose(result, expected)
            else:
                angle = 0.986
                point = gs.array([
                    [gs.cos(angle), -gs.sin(angle)],
                    [gs.sin(angle), gs.cos(angle)]])

                inv_point = group.inverse(point)
                # Compose transformation by its inverse on the right
                # Expect the group identity
                result = group.compose(point, inv_point)
                expected = helper.to_matrix(group.identity)
                self.assertAllClose(result, expected)

                # Compose transformation by its inverse on the left
                # Expect the group identity
                result = group.compose(inv_point, point)
                expected = helper.to_matrix(group.identity)
                self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_compose_vectorization(self):
        for point_type in ('vector', 'matrix'):
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
                        gs.shape(result), (n_samples, group.dimension))
                if point_type == 'matrix':
                    self.assertAllClose(
                        gs.shape(result), (n_samples, n, n))

                result = group.compose(n_points_a, one_point)
                if point_type == 'vector':
                    self.assertAllClose(
                        gs.shape(result), (n_samples, group.dimension))
                if point_type == 'matrix':
                    self.assertAllClose(
                        gs.shape(result), (n_samples, n, n))

                result = group.compose(n_points_a, n_points_b)
                if point_type == 'vector':
                    self.assertAllClose(
                        gs.shape(result), (n_samples, group.dimension))
                if point_type == 'matrix':
                    self.assertAllClose(
                        gs.shape(result), (n_samples, n, n))

    @geomstats.tests.np_only
    def test_inverse_vectorization(self):
        for n in self.n_seq:
            group = self.so[n]

            n_samples = self.n_samples
            points = group.random_uniform(n_samples=n_samples)
            result = group.inverse(points)

            if n == 3:
                self.assertAllClose(
                    gs.shape(result), (n_samples, group.dimension))
            else:
                self.assertAllClose(
                    gs.shape(result), (n_samples, n, n))

    @geomstats.tests.np_only
    def test_left_jacobian_through_its_determinant(self):
        n = 3
        group = self.so[n]

        for angle_type in self.elements[3]:
            point = self.elements[3][angle_type]
            jacobian = group.jacobian_translation(point=point,
                                                  left_or_right='left')
            result = gs.linalg.det(jacobian)
            point = group.regularize(point)
            angle = gs.linalg.norm(point)
            if angle_type in ['with_angle_0',
                              'with_angle_close_0',
                              'with_angle_2pi',
                              'with_angle_close_2pi_high']:
                expected = 1. + angle ** 2 / 12. + angle ** 4 / 240.
            else:
                expected = angle ** 2 / (4 * gs.sin(angle / 2) ** 2)

            expected = gs.to_ndarray(expected, to_ndim=1)
            self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_left_jacobian_vectorization(self):
        n = 3
        group = self.so[n]

        n_samples = self.n_samples
        points = group.random_uniform(n_samples=n_samples)
        jacobians = group.jacobian_translation(point=points,
                                               left_or_right='left')
        self.assertAllClose(
            gs.shape(jacobians),
            (n_samples, group.dimension, group.dimension))

    @geomstats.tests.np_only
    def test_exp(self):
        """
        The Riemannian exp and log are inverse functions of each other.
        This test is the inverse of test_log's.
        """
        n = 3
        group = self.so[n]

        metric = self.metrics_all[3]['canonical']
        theta = gs.pi / 5.
        rot_vec_base_point = theta / gs.sqrt(3.) * gs.array([1., 1., 1.])
        # Note: the rotation vector for the reference point
        # needs to be regularized.

        # 1: Exponential of 0 gives the reference point
        rot_vec_1 = gs.array([0., 0., 0.])
        expected = helper.to_vector(rot_vec_base_point)

        result = metric.exp(base_point=rot_vec_base_point,
                            tangent_vec=rot_vec_1)
        self.assertAllClose(result, expected)

        # 2: General case - computed manually
        rot_vec_2 = gs.pi / 4 * gs.array([1., 0., 0.])
        phi = (gs.pi / 10) / (gs.tan(gs.pi / 10))
        skew = gs.array([[0., -1., 1.],
                         [1., 0., -1.],
                         [-1., 1., 0.]])
        jacobian = (phi * gs.eye(3)
                    + (1 - phi) / 3 * gs.ones([3, 3])
                    + gs.pi / (10 * gs.sqrt(3.)) * skew)
        inv_jacobian = gs.linalg.inv(jacobian)
        expected = group.compose(rot_vec_base_point,
                                 gs.dot(inv_jacobian, rot_vec_2))

        result = metric.exp(base_point=rot_vec_base_point,
                            tangent_vec=rot_vec_2)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_exp_vectorization(self):
        n = 3
        group = self.so[n]

        n_samples = self.n_samples
        for metric_type in self.metrics[3]:
            metric = self.metrics[3][metric_type]

            one_tangent_vec = group.random_uniform(n_samples=1)
            one_base_point = group.random_uniform(n_samples=1)
            n_tangent_vec = group.random_uniform(n_samples=n_samples)
            n_base_point = group.random_uniform(n_samples=n_samples)

            # Test with the 1 base point, and n tangent vecs
            result = metric.exp(n_tangent_vec, one_base_point)
            self.assertAllClose(
                gs.shape(result), (n_samples, group.dimension))

            # Test with the several base point, and one tangent vec
            result = metric.exp(one_tangent_vec, n_base_point)
            self.assertAllClose(
                gs.shape(result), (n_samples, group.dimension))

            # Test with the same number n of base point and n tangent vec
            result = metric.exp(n_tangent_vec, n_base_point)
            self.assertAllClose(
                gs.shape(result), (n_samples, group.dimension))

    @geomstats.tests.np_only
    def test_log(self):
        """
        The Riemannian exp and log are inverse functions of each other.
        This test is the inverse of test_exp's.
        """
        n = 3
        group = self.so[n]

        metric = self.metrics_all[3]['canonical']
        theta = gs.pi / 5.
        rot_vec_base_point = theta / gs.sqrt(3.) * gs.array([1., 1., 1.])
        # Note: the rotation vector for the reference point
        # needs to be regularized.

        # The Logarithm of a point at itself gives 0.
        rot_vec_1 = rot_vec_base_point
        expected = gs.array([[0., 0., 0.]])
        result = metric.log(base_point=rot_vec_base_point,
                            point=rot_vec_1)
        self.assertAllClose(result, expected)

        # General case: this is the inverse test of test 1 for riemannian exp
        expected = gs.pi / 4 * gs.array([1., 0., 0.])
        phi = (gs.pi / 10) / (gs.tan(gs.pi / 10))
        skew = gs.array([[0., -1., 1.],
                         [1., 0., -1.],
                         [-1., 1., 0.]])
        jacobian = (phi * gs.eye(3)
                    + (1 - phi) / 3 * gs.ones([3, 3])
                    + gs.pi / (10 * gs.sqrt(3.)) * skew)
        inv_jacobian = gs.linalg.inv(jacobian)
        aux = gs.dot(inv_jacobian, expected)
        rot_vec_2 = group.compose(rot_vec_base_point,
                                  aux)

        result = metric.log(base_point=rot_vec_base_point,
                            point=rot_vec_2)

        expected = helper.to_vector(expected)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_log_vectorization(self):
        n = 3
        group = self.so[n]

        n_samples = self.n_samples
        for metric_type in self.metrics[3]:
            metric = self.metrics[3][metric_type]

            one_point = group.random_uniform(n_samples=1)
            one_base_point = group.random_uniform(n_samples=1)
            n_point = group.random_uniform(n_samples=n_samples)
            n_base_point = group.random_uniform(n_samples=n_samples)

            # Test with the 1 base point, and several different points
            result = metric.log(n_point, one_base_point)
            self.assertAllClose(
                gs.shape(result), (n_samples, group.dimension))

            # Test with the several base point, and 1 point
            result = metric.log(one_point, n_base_point)
            self.assertAllClose(
                gs.shape(result), (n_samples, group.dimension))

            # Test with the same number n of base point and point
            result = metric.log(n_point, n_base_point)
            self.assertAllClose(
                gs.shape(result), (n_samples, group.dimension))

    @geomstats.tests.np_only
    def test_exp_from_identity_vectorization(self):
        n = 3
        group = self.so[n]

        n_samples = self.n_samples
        metric = self.metrics_all[3]['canonical']

        tangent_vecs = group.random_uniform(n_samples=n_samples)
        result = metric.exp_from_identity(tangent_vecs)

        self.assertAllClose(
            gs.shape(result), (n_samples, group.dimension))

    @geomstats.tests.np_only
    def test_log_from_identity_vectorization(self):
        n = 3
        group = self.so[n]

        n_samples = self.n_samples
        metric = self.metrics_all[3]['canonical']

        points = group.random_uniform(n_samples=n_samples)
        result = metric.log_from_identity(points)

        self.assertAllClose(
            gs.shape(result), (n_samples, group.dimension))

    @geomstats.tests.np_only
    def test_exp_then_log_from_identity(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        n = 3
        group = self.so[n]

        for metric_type in self.metrics[3]:
            for angle_type in self.elements[3]:
                if angle_type in self.angles_close_to_pi[3]:
                    continue

                metric = self.metrics[3][metric_type]
                tangent_vec = self.elements[3][angle_type]

                result = helper.exp_then_log_from_identity(metric, tangent_vec)

                reg_vec = group.regularize_tangent_vec_at_identity(
                    tangent_vec=tangent_vec, metric=metric)
                expected = reg_vec

                self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_exp_then_log_from_identity_with_angles_close_to_pi(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        n = 3
        group = self.so[n]

        angle_types = self.angles_close_to_pi[3]

        for metric_type in self.metrics[3]:
            for angle_type in angle_types:

                metric = self.metrics[3][metric_type]
                tangent_vec = self.elements[3][angle_type]

                result = helper.exp_then_log_from_identity(metric, tangent_vec)

                expected = group.regularize_tangent_vec_at_identity(
                    tangent_vec=tangent_vec, metric=metric)
                inv_expected = - expected
                self.assertTrue(
                    gs.eval(gs.allclose(result, expected))
                    or gs.eval(gs.allclose(result, inv_expected)))

    @geomstats.tests.np_only
    def test_log_then_exp_from_identity(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """

        n = 3
        group = self.so[n]

        for metric_type in self.metrics[3]:
            for angle_type in self.elements[3]:
                if angle_type in self.angles_close_to_pi[3]:
                    continue

                metric = self.metrics[3][metric_type]
                point = self.elements[3][angle_type]

                result = helper.log_then_exp_from_identity(metric, point)
                expected = group.regularize(point)

                self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_log_then_exp_from_identity_with_angles_close_to_pi(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        n = 3
        group = self.so[n]

        angle_types = self.angles_close_to_pi[3]

        for metric_type in self.metrics[3]:
            for angle_type in angle_types:

                metric = self.metrics[3][metric_type]
                point = self.elements[3][angle_type]

                result = helper.log_then_exp_from_identity(metric, point)
                expected = group.regularize(point)
                inv_expected = - expected

                self.assertTrue(
                    gs.eval(gs.allclose(result, expected))
                    or gs.eval(gs.allclose(result, inv_expected)))

    @geomstats.tests.np_only
    def test_exp_then_log(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        n = 3
        group = self.so[n]

        for metric_type in self.metrics[3]:
            for angle_type in self.elements[3]:
                if angle_type in self.angles_close_to_pi[3]:
                    continue
                for angle_type_base in self.elements[3]:

                    metric = self.metrics[3][metric_type]
                    tangent_vec = self.elements[3][angle_type]
                    base_point = self.elements[3][angle_type_base]

                    result = helper.exp_then_log(metric=metric,
                                                 tangent_vec=tangent_vec,
                                                 base_point=base_point)

                    reg_tangent_vec = group.regularize_tangent_vec(
                        tangent_vec=tangent_vec,
                        base_point=base_point,
                        metric=metric)
                    expected = reg_tangent_vec

                    self.assertAllClose(result, expected, atol=1e-4)

    @geomstats.tests.np_only
    def test_exp_then_log_with_angles_close_to_pi(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        n = 3
        group = self.so[n]

        angle_types = self.angles_close_to_pi[3]
        for metric_type in self.metrics[3]:
            for angle_type in angle_types:
                for angle_type_base in self.elements[3]:

                    metric = self.metrics[3][metric_type]
                    tangent_vec = self.elements[3][angle_type]
                    base_point = self.elements[3][angle_type_base]

                    result = helper.exp_then_log(metric=metric,
                                                 tangent_vec=tangent_vec,
                                                 base_point=base_point)

                    reg_tangent_vec = group.regularize_tangent_vec(
                        tangent_vec=tangent_vec,
                        base_point=base_point,
                        metric=metric)
                    expected = reg_tangent_vec
                    inv_expected = - expected

                    self.assertTrue(
                        gs.eval(gs.allclose(
                            result, expected, atol=1e-5))
                        or gs.eval(gs.allclose(
                            result, inv_expected, atol=1e-5)))

    @geomstats.tests.np_only
    def test_log_then_exp(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """

        n = 3
        group = self.so[n]

        for metric_type in self.metrics[3]:
            for angle_type in self.elements[3]:
                if angle_type in self.angles_close_to_pi[3]:
                    continue
                for angle_type_base in self.elements[3]:
                    if angle_type_base in self.angles_close_to_pi[3]:
                        continue
                    metric = self.metrics[3][metric_type]
                    point = self.elements[3][angle_type]
                    base_point = self.elements[3][angle_type_base]

                    result = helper.log_then_exp(metric=metric,
                                                 base_point=base_point,
                                                 point=point)

                    expected = group.regularize(point)
                    inv_expected = - expected

                    self.assertTrue(
                        gs.eval(gs.allclose(result, expected))
                        or gs.eval(gs.allclose(result, inv_expected)))

    @geomstats.tests.np_only
    def test_log_then_exp_with_angles_close_to_pi(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        n = 3
        group = self.so[n]

        angle_types = self.angles_close_to_pi[3]
        for metric_type in self.metrics[3]:
            for angle_type in angle_types:
                for angle_type_base in self.elements[3]:
                    metric = self.metrics[3][metric_type]
                    point = self.elements[3][angle_type]
                    base_point = self.elements[3][angle_type_base]

                    result = helper.log_then_exp(metric=metric,
                                                 base_point=base_point,
                                                 point=point)

                    expected = group.regularize(point)
                    inv_expected = - expected

                    self.assertTrue(
                        gs.eval(gs.allclose(result, expected, atol=1e-5))
                        or gs.eval(gs.allclose(
                            result, inv_expected, atol=1e-5)))

    @geomstats.tests.np_only
    def test_group_exp_from_identity_coincides_with_expm(self):
        """Test exponentials."""
        # FIXME: Problem in shapes
        # for n in self.n_seq:
        #     group = self.so[n]
        #     dim = int(n * (n - 1) / 2)

        #     normal_rv = gs.random.normal(size=dim)
        #     tangent_sample = gs.zeros((n, n))
        #     tangent_sample[gs.triu_indices(n, k=1)] = normal_rv
        #     tangent_sample = tangent_sample - gs.transpose(tangent_sample)
        #     result = gs.linalg.expm(tangent_sample)
        #     expected = group.exp_from_identity(
        #         tangent_sample, point_type='matrix')

        #     self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_group_exp_from_identity_coincides_with_expm_for_high_dims(self):
        for n in [4, 5, 6, 7, 8, 9, 10]:
            group = SpecialOrthogonal(n=n)
            dim = int(n * (n - 1) / 2)

            normal_rv = gs.random.normal(size=dim)
            tangent_sample = gs.zeros((n, n))
            tangent_sample[gs.triu_indices(n, k=1)] = normal_rv
            tangent_sample = tangent_sample - gs.transpose(tangent_sample)

            result = gs.reshape(
                group.exp_from_identity(
                    tangent_sample, point_type='matrix'), (n, n))

            expected = gs.linalg.expm(tangent_sample)

            self.assertAllClose(result, expected)

    def test_group_exp_from_identity_vectorization(self):
        n = 3
        group = self.so[n]

        n_samples = self.n_samples
        tangent_vecs = group.random_uniform(n_samples=n_samples)
        result = group.exp_from_identity(tangent_vecs)

        self.assertAllClose(
            gs.shape(result), (n_samples, group.dimension))

    def test_group_log_from_identity_vectorization(self):
        n = 3
        group = self.so[n]

        n_samples = self.n_samples
        points = group.random_uniform(n_samples=n_samples)
        result = group.log_from_identity(points)

        self.assertAllClose(
            gs.shape(result), (n_samples, group.dimension))

    @geomstats.tests.np_only
    def test_group_exp_vectorization(self):
        n = 3
        group = self.so[n]

        n_samples = self.n_samples
        # Test with the 1 base_point, and several different tangent_vecs
        tangent_vecs = group.random_uniform(n_samples=n_samples)
        base_point = group.random_uniform(n_samples=1)
        result = group.exp(tangent_vecs, base_point)

        self.assertAllClose(
            gs.shape(result), (n_samples, group.dimension))

        # Test with the same number of base_points and tangent_vecs
        tangent_vecs = group.random_uniform(n_samples=n_samples)
        base_points = group.random_uniform(n_samples=n_samples)
        result = group.exp(tangent_vecs, base_points)

        self.assertAllClose(
            gs.shape(result), (n_samples, group.dimension))

        # Test with the several base_points, and 1 tangent_vec
        tangent_vec = group.random_uniform(n_samples=1)
        base_points = group.random_uniform(n_samples=n_samples)
        result = group.exp(tangent_vec, base_points)

        self.assertAllClose(
            gs.shape(result), (n_samples, group.dimension))

    @geomstats.tests.np_only
    def test_group_log_vectorization(self):
        n = 3
        group = self.so[n]

        n_samples = self.n_samples
        # Test with the 1 base point, and several different points
        points = group.random_uniform(n_samples=n_samples)
        base_point = group.random_uniform(n_samples=1)
        result = group.log(points, base_point)

        self.assertAllClose(
            gs.shape(result), (n_samples, group.dimension))

        # Test with the same number of base points and points
        points = group.random_uniform(n_samples=n_samples)
        base_points = group.random_uniform(n_samples=n_samples)
        result = group.log(points, base_points)

        self.assertAllClose(
            gs.shape(result), (n_samples, group.dimension))

        # Test with the several base points, and 1 point
        point = group.random_uniform(n_samples=1)
        base_points = group.random_uniform(n_samples=n_samples)
        result = group.log(point, base_points)

        self.assertAllClose(
            gs.shape(result), (n_samples, group.dimension))

    def test_group_exp_then_log_from_identity(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        n = 3
        group = self.so[n]

        for angle_type in self.elements[3]:
            if angle_type in self.angles_close_to_pi[3]:
                continue
            tangent_vec = self.elements[3][angle_type]
            result = helper.group_exp_then_log_from_identity(
                group=group, tangent_vec=tangent_vec)
            expected = group.regularize(tangent_vec)
            self.assertAllClose(result, expected)

    def test_group_exp_then_log_from_identity_with_angles_close_to_pi(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        n = 3
        group = self.so[n]

        angle_types = self.angles_close_to_pi[3]
        for angle_type in angle_types:
            tangent_vec = self.elements[3][angle_type]
            result = helper.group_exp_then_log_from_identity(
                group=group,
                tangent_vec=tangent_vec)
            expected = group.regularize(tangent_vec)
            inv_expected = - expected

            with self.session():
                self.assertTrue(
                    gs.eval(gs.allclose(result, expected))
                    or gs.eval(gs.allclose(result, inv_expected)))

    def test_group_log_then_exp_from_identity(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        n = 3
        group = self.so[n]

        for angle_type in self.elements[3]:
            point = self.elements[3][angle_type]
            result = helper.group_log_then_exp_from_identity(
                group=group,
                point=point)
            expected = group.regularize(point)
            self.assertAllClose(result, expected)

    def test_group_log_then_exp_from_identity_with_angles_close_to_pi(self):
        """
        Test that the group exponential
        and the group logarithm are inverse.
        Expect their composition to give the identity function.
        """
        n = 3
        group = self.so[n]

        angle_types = self.angles_close_to_pi[3]
        for angle_type in angle_types:
            point = self.elements[3][angle_type]
            result = helper.group_log_then_exp_from_identity(
                group=group,
                point=point)
            expected = group.regularize(point)
            inv_expected = - expected

            with self.session():
                self.assertTrue(
                    gs.eval(gs.allclose(result, expected))
                    or gs.eval(gs.allclose(result, inv_expected)))

    @geomstats.tests.np_only
    def test_group_exp_then_log(self):
        """
        This tests that the composition of
        log and exp gives identity.

        """
        n = 3
        group = self.so[n]

        for angle_type in self.elements[3]:
            if angle_type in self.angles_close_to_pi[3]:
                continue
            for angle_type_base in self.elements[3]:
                tangent_vec = self.elements[3][angle_type]
                base_point = self.elements[3][angle_type_base]

                result = helper.group_exp_then_log(
                    group=group,
                    tangent_vec=tangent_vec,
                    base_point=base_point)

                # TODO(nina): Investigate what it means to regularize
                # the tangent vector when there is no metric.
                metric = group.left_canonical_metric
                expected = group.regularize_tangent_vec(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    metric=metric)

                self.assertAllClose(result, expected, atol=1e-6)

    @geomstats.tests.np_only
    def test_group_exp_then_log_with_angles_close_to_pi(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        n = 3
        group = self.so[n]

        angle_types = self.angles_close_to_pi[3]
        for angle_type in angle_types:
            for angle_type_base in self.elements[3]:
                tangent_vec = self.elements[3][angle_type]
                base_point = self.elements[3][angle_type_base]

                result = helper.group_exp_then_log(
                    group=group,
                    tangent_vec=tangent_vec,
                    base_point=base_point)

                # TODO(nina): Investigate what it means to regularize
                # the tangent vector when there is no metric.
                metric = group.left_canonical_metric
                reg_tangent_vec = group.regularize_tangent_vec(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    metric=metric)
                expected = reg_tangent_vec
                inv_expected = - expected

                self.assertTrue(
                    gs.eval(gs.allclose(result, expected, atol=1e-5))
                    or gs.eval(gs.allclose(result, inv_expected, atol=1e-5)))

    @geomstats.tests.np_only
    def test_group_log_then_exp(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """

        n = 3
        group = self.so[n]

        for angle_type in self.elements[3]:
            if angle_type in self.angles_close_to_pi[3]:
                continue
            for angle_type_base in self.elements[3]:
                if angle_type_base in self.angles_close_to_pi[3]:
                    continue
                point = self.elements[3][angle_type]
                base_point = self.elements[3][angle_type_base]

                result = helper.group_log_then_exp(
                    group=group,
                    point=point,
                    base_point=base_point)
                expected = group.regularize(point)

                self.assertAllClose(result, expected, atol=ATOL)

    @geomstats.tests.np_only
    def test_group_log_then_exp_with_angles_close_to_pi(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        n = 3
        group = self.so[n]

        angle_types = self.angles_close_to_pi[3]
        for angle_type in angle_types:
            for angle_type_base in self.elements[3]:
                point = self.elements[3][angle_type]
                base_point = self.elements[3][angle_type_base]

                result = helper.group_log_then_exp(
                    group=group,
                    point=point,
                    base_point=base_point)
                expected = group.regularize(point)
                inv_expected = - expected

                self.assertTrue(
                    gs.allclose(result, expected, atol=1e-5)
                    or gs.allclose(result, inv_expected, atol=1e-5))

    @geomstats.tests.np_only
    def test_group_exponential_barycenter(self):
        n = 3
        group = self.so[n]

        rot_vec = group.random_uniform()
        points = gs.vstack([rot_vec, rot_vec])
        result = group.exponential_barycenter(
            points=points)
        expected = rot_vec
        self.assertAllClose(result, expected)

        rot_vec = group.random_uniform()
        points = gs.vstack([rot_vec, rot_vec])
        weights = gs.array([1., 2.])
        result = group.exponential_barycenter(
            points=points,
            weights=weights)
        expected = rot_vec
        self.assertAllClose(result, expected)

        points = gs.vstack([rot_vec, rot_vec])
        weights = gs.array([1., 2.])
        group_bar = group.exponential_barycenter(
            points=points, weights=weights)
        result = group.belongs(group_bar)
        expected = gs.array([[True]])

        self.assertTrue(result, expected)

    @geomstats.tests.np_only
    def test_squared_dist_is_symmetric(self):
        n = 3
        group = self.so[n]

        for metric in self.metrics[3].values():
            for angle_type_1 in self.elements[3]:
                for angle_type_2 in self.elements[3]:
                    point_1 = self.elements[3][angle_type_1]
                    point_2 = self.elements[3][angle_type_2]
                    point_1 = group.regularize(point_1)
                    point_2 = group.regularize(point_2)

                    sq_dist_1_2 = gs.mod(
                        metric.squared_dist(point_1, point_2) + 1e-4,
                        gs.pi**2)
                    sq_dist_2_1 = gs.mod(
                        metric.squared_dist(point_2, point_1) + 1e-4,
                        gs.pi**2)

                    self.assertAllClose(sq_dist_1_2, sq_dist_2_1, atol=1e-4)

    @geomstats.tests.np_only
    def test_squared_dist_is_less_than_squared_pi(self):
        """
        This test only concerns the canonical metric.
        For other metrics, the scaling factor can give
        distances above pi.
        """
        n = 3
        group = self.so[n]

        metric = self.metrics_all[3]['canonical']
        for angle_type_1 in self.elements[3]:
            for angle_type_2 in self.elements[3]:
                point_1 = self.elements[3][angle_type_1]
                point_2 = self.elements[3][angle_type_2]
                point_1 = group.regularize(point_1)
                point_2 = group.regularize(point_2)

                sq_dist = metric.squared_dist(point_1, point_2)
                diff = sq_dist - gs.pi ** 2
                self.assertTrue(diff <= 0 or abs(diff) < EPSILON,
                                'sq_dist = {}'.format(sq_dist))

    @geomstats.tests.np_only
    def test_squared_dist_vectorization(self):
        n = 3
        group = self.so[n]

        n_samples = self.n_samples
        for metric_type in self.metrics[3]:
            metric = self.metrics[3][metric_type]
            point_id = group.identity

            one_point_1 = group.random_uniform(n_samples=1)
            one_point_2 = group.random_uniform(n_samples=1)
            one_point_1 = group.regularize(one_point_1)
            one_point_2 = group.regularize(one_point_2)

            n_point_1 = group.random_uniform(n_samples=n_samples)
            n_point_2 = group.random_uniform(n_samples=n_samples)
            n_point_1 = group.regularize(n_point_1)
            n_point_2 = group.regularize(n_point_2)

            # Identity and n points 2
            result = metric.squared_dist(point_id, n_point_2)
            self.assertAllClose(gs.shape(result), (n_samples, 1))

            # n points 1 and identity
            result = metric.squared_dist(n_point_1, point_id)
            self.assertAllClose(gs.shape(result), (n_samples, 1))

            # one point 1 and n points 2
            result = metric.squared_dist(one_point_1, n_point_2)
            self.assertAllClose(gs.shape(result), (n_samples, 1))

            # n points 1 and one point 2
            result = metric.squared_dist(n_point_1, one_point_2)
            self.assertAllClose(gs.shape(result), (n_samples, 1))

            # n points 1 and n points 2
            result = metric.squared_dist(n_point_1, n_point_2)
            self.assertAllClose(gs.shape(result), (n_samples, 1))

    @geomstats.tests.np_only
    def test_dist_vectorization(self):
        n = 3
        group = self.so[n]

        n_samples = self.n_samples
        for metric_type in self.metrics[3]:
            metric = self.metrics[3][metric_type]
            point_id = group.identity

            one_point_1 = group.random_uniform(n_samples=1)
            one_point_2 = group.random_uniform(n_samples=1)
            one_point_1 = group.regularize(one_point_1)
            one_point_2 = group.regularize(one_point_2)

            n_point_1 = group.random_uniform(n_samples=n_samples)
            n_point_2 = group.random_uniform(n_samples=n_samples)
            n_point_1 = group.regularize(n_point_1)
            n_point_2 = group.regularize(n_point_2)

            # Identity and n points 2
            result = metric.dist(point_id, n_point_2)
            self.assertAllClose(gs.shape(result), (n_samples, 1))

            # n points 1 and identity
            result = metric.dist(n_point_1, point_id)
            self.assertAllClose(gs.shape(result), (n_samples, 1))

            # one point 1 and n points 2
            result = metric.dist(one_point_1, n_point_2)
            self.assertAllClose(gs.shape(result), (n_samples, 1))

            # n points 1 and one point 2
            result = metric.dist(n_point_1, one_point_2)
            self.assertAllClose(gs.shape(result), (n_samples, 1))

            # n points 1 and n points 2
            result = metric.dist(n_point_1, n_point_2)
            self.assertAllClose(gs.shape(result), (n_samples, 1))

    @geomstats.tests.np_only
    def test_geodesic_and_belongs(self):
        n = 3
        group = self.so[n]

        initial_point = group.random_uniform()
        initial_tangent_vec = gs.array([2., 0., -1.])
        metric = self.metrics_all[3]['canonical']
        geodesic = metric.geodesic(initial_point=initial_point,
                                   initial_tangent_vec=initial_tangent_vec)

        t = gs.linspace(start=0., stop=1., num=100)
        points = geodesic(t)
        result = gs.all(group.belongs(points))
        expected = True
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_geodesic_subsample(self):
        """Test geodesic."""
        # FIXME
        # n = 3
        # group = self.so[n]

        # initial_point = group.random_uniform()
        # initial_tangent_vec = gs.array([1., 1., 1.])
        # metric = self.metrics_all[n]['canonical']
        # geodesic = metric.geodesic(initial_point=initial_point,
        #                            initial_tangent_vec=initial_tangent_vec)
        # n_steps = 100
        # t = gs.linspace(start=0., stop=1., num=n_steps+1)
        # points = geodesic(t)

        # tangent_vec_step = initial_tangent_vec / n_steps
        # for i in range(n_steps+1):
        #     point_step = metric.exp(tangent_vec=i * tangent_vec_step,
        #                             base_point=initial_point)
        #     self.assertTrue(gs.allclose(point_step, points[i]))

    def test_lie_bracket_at_identity(self):
        dim = 3
        space = self.so[dim]
        base_point = gs.eye(dim)
        first_tan = gs.array([
            [0., -1., 0.],
            [1., 0., 0.],
            [0., 0., 0.]])
        second_tan = first_tan

        result = space.lie_bracket(first_tan, second_tan, base_point)
        expected = gs.zeros((1, dim, dim))

        self.assertAllClose(result, expected)

        first_tan = gs.array([
            [0., -1., 0.],
            [1., 0., 0.],
            [0., 0., 0.]])
        second_tan = gs.array([
            [0., 0., -1.],
            [0., 0., 0.],
            [1., 0., 0.]])

        result = space.lie_bracket(first_tan, second_tan, base_point)
        expected = gs.array([[
            [0., 0., 0.],
            [0., 0., -1.],
            [0., 1., 0.]]])

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_lie_bracket_vectorization(self):
        dim = 3
        space = self.so[dim]

        base_point = gs.array([gs.eye(dim), gs.eye(dim)])
        first_tan = gs.array([
            [[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]],
            [[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]],
        ])
        second_tan = gs.array([
            [[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]],
            [[0., 0., -1.], [0., 0., 0.], [1., 0., 0.]]
        ])

        result = space.lie_bracket(first_tan, second_tan, base_point)
        expected = gs.array([
            gs.zeros((dim, dim)),
            [[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]]
        ])

        self.assertAllClose(result, expected)

    def test_lie_bracket_at_non_identity(self):
        dim = 3
        space = self.so[dim]

        base_point = gs.array([
            [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]])
        first_tan = gs.matmul(
            base_point,
            gs.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])
        )
        second_tan = gs.matmul(
            base_point,
            gs.array([[0., 0., -1.], [0., 0., 0.], [1., 0., 0.]])
        )

        result = space.lie_bracket(first_tan, second_tan, base_point)
        expected = gs.matmul(
            base_point,
            gs.array([[[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]]]))

        self.assertAllClose(result, expected)
