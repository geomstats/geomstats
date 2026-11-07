import geomstats.backend as gs
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.lie_group import (
    LieGroupTestCase,
    MatrixLieGroupTestCase,
)
from geomstats.test_cases.geometry.mixins import ProjectionTestCaseMixins


class _SpecialOrthogonalTestCaseMixins:
    def test_rotation_vector_from_matrix(self, rot_mat, expected, atol=gs.atol):
        """Check rotation vector from rotation matrix.

        Parameters
        ----------
        rot_mat : array-like, shape=[..., n, n]
            Rotation matrix.
        expected : array-like, shape=[..., dim]
            Expected rotation vector.
        atol : float
            Absolute tolerance.
        """
        rot_vec = self.space.rotation_vector_from_matrix(rot_mat)
        self.assertAllClose(rot_vec, expected, atol=atol)

    def test_matrix_from_rotation_vector(self, rot_vec, expected, atol=gs.atol):
        """Check rotation matrix from rotation vector.

        Parameters
        ----------
        rot_vec : array-like, shape=[..., dim]
            Rotation vector.
        expected : array-like, shape=[..., n, n]
            Expected rotation matrix.
        atol : float
            Absolute tolerance.
        """
        rot_mat = self.space.matrix_from_rotation_vector(rot_vec)
        self.assertAllClose(rot_mat, expected, atol=atol)

    def test_rotation_vector_from_matrix_after_matrix_from_rotation_vector(
        self, rot_vec, atol=gs.atol
    ):
        """Check rotation vector after rotation matrix gives back the vector.

        Parameters
        ----------
        rot_vec : array-like, shape=[..., dim]
            Rotation vector.
        atol : float
            Absolute tolerance.
        """
        rot_mat = self.space.matrix_from_rotation_vector(rot_vec)
        rot_vec_ = self.space.rotation_vector_from_matrix(rot_mat)
        self.assertAllClose(rot_vec_, rot_vec, atol=atol)

    def test_matrix_from_rotation_vector_after_rotation_vector_from_matrix(
        self, rot_mat, atol=gs.atol
    ):
        """Check rotation matrix after rotation vector gives back the matrix.

        Parameters
        ----------
        rot_mat : array-like, shape=[..., n, n]
            Rotation matrix.
        atol : float
            Absolute tolerance.
        """
        rot_vec = self.space.rotation_vector_from_matrix(rot_mat)
        rot_mat_ = self.space.matrix_from_rotation_vector(rot_vec)
        self.assertAllClose(rot_mat_, rot_mat, atol=atol)


class SpecialOrthogonalMatricesTestCase(
    _SpecialOrthogonalTestCaseMixins, MatrixLieGroupTestCase, LevelSetTestCase
):
    def test_to_tangent(self, vector, base_point, expected, atol=gs.atol):
        """Check projection to the tangent space.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Matrix.
        base_point : array-like, shape=[..., n, n]
            Rotation matrix. If None, the identity.
        expected : array-like, shape=[..., n, n]
            Expected tangent vector at `base_point`.
        atol : float
            Absolute tolerance.
        """
        res = self.space.to_tangent(vector, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_are_antipodals(self, rotation_mat1, rotation_mat2, expected, atol=gs.atol):
        """Check whether rotation matrices are antipodal.

        Parameters
        ----------
        rotation_mat1 : array-like, shape=[..., n, n]
            Rotation matrix.
        rotation_mat2 : array-like, shape=[..., n, n]
            Rotation matrix.
        expected : array-like, shape=[...]
            Expected boolean.
        atol : float
            Absolute tolerance.
        """
        res = self.space.are_antipodals(rotation_mat1, rotation_mat2)
        self.assertAllClose(res, expected, atol=atol)


class SpecialOrthogonalVectorsTestCase(
    ProjectionTestCaseMixins, _SpecialOrthogonalTestCaseMixins, LieGroupTestCase
):
    pass


class SpecialOrthogonal3VectorsTestCase(SpecialOrthogonalVectorsTestCase):
    def _assert_quaternion(self, quaternion_, quaternion, atol):
        self.assertAllClose(gs.abs(quaternion_), gs.abs(quaternion), atol=atol)

    def _assert_tait_bryan_angles(self, angles_, angles, extrinsic, zyx, atol):
        try:
            self.assertAllClose(angles_, angles, atol=atol)
        except AssertionError:
            mat_ = self.space.matrix_from_tait_bryan_angles(
                angles_, extrinsic=extrinsic, zyx=zyx
            )
            mat = self.space.matrix_from_tait_bryan_angles(
                angles, extrinsic=extrinsic, zyx=zyx
            )
            self.assertAllClose(mat_, mat, atol=atol)

    def test_quaternion_from_matrix(self, rot_mat, expected, atol=gs.atol):
        """Check quaternion from rotation matrix.

        Parameters
        ----------
        rot_mat : array-like, shape=[..., 3, 3]
            Rotation matrix.
        expected : array-like, shape=[..., 4]
            Expected quaternion.
        atol : float
            Absolute tolerance.
        """
        res = self.space.quaternion_from_matrix(rot_mat)
        self.assertAllClose(res, expected, atol=atol)

    def test_matrix_from_quaternion(self, quaternion, expected, atol=gs.atol):
        """Check rotation matrix from quaternion.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]
            Quaternion.
        expected : array-like, shape=[..., 3, 3]
            Expected rotation matrix.
        atol : float
            Absolute tolerance.
        """
        res = self.space.matrix_from_quaternion(quaternion)
        self.assertAllClose(res, expected, atol=atol)

    def test_quaternion_from_rotation_vector(self, rot_vec, expected, atol=gs.atol):
        """Check quaternion from rotation vector.

        Parameters
        ----------
        rot_vec : array-like, shape=[..., 3]
            Rotation vector.
        expected : array-like, shape=[..., 4]
            Expected quaternion.
        atol : float
            Absolute tolerance.
        """
        res = self.space.quaternion_from_rotation_vector(rot_vec)
        self.assertAllClose(res, expected, atol=atol)

    def test_rotation_vector_from_quaternion(self, quaternion, expected, atol=gs.atol):
        """Check rotation vector from quaternion.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]
            Quaternion.
        expected : array-like, shape=[..., 3]
            Expected rotation vector.
        atol : float
            Absolute tolerance.
        """
        res = self.space.rotation_vector_from_quaternion(quaternion)
        self.assertAllClose(res, expected, atol=atol)

    def test_matrix_from_tait_bryan_angles(
        self, tait_bryan_angles, extrinsic, zyx, expected, atol=gs.atol
    ):
        """Check rotation matrix from Tait-Bryan angles.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]
            Tait-Bryan angles.
        extrinsic : bool
            Whether the angles are extrinsic or intrinsic.
        zyx : bool
            Whether the angles are in zyx or xyz order.
        expected : array-like, shape=[..., 3, 3]
            Expected rotation matrix.
        atol : float
            Absolute tolerance.
        """
        res = self.space.matrix_from_tait_bryan_angles(
            tait_bryan_angles, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_tait_bryan_angles_from_matrix(
        self, rot_mat, extrinsic, zyx, expected, atol=gs.atol
    ):
        """Check Tait-Bryan angles from rotation matrix.

        Parameters
        ----------
        rot_mat : array-like, shape=[..., 3, 3]
            Rotation matrix.
        extrinsic : bool
            Whether the angles are extrinsic or intrinsic.
        zyx : bool
            Whether the angles are in zyx or xyz order.
        expected : array-like, shape=[..., 3]
            Expected Tait-Bryan angles.
        atol : float
            Absolute tolerance.
        """
        res = self.space.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_quaternion_from_tait_bryan_angles(
        self, tait_bryan_angles, extrinsic, zyx, expected, atol=gs.atol
    ):
        """Check quaternion from Tait-Bryan angles.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]
            Tait-Bryan angles.
        extrinsic : bool
            Whether the angles are extrinsic or intrinsic.
        zyx : bool
            Whether the angles are in zyx or xyz order.
        expected : array-like, shape=[..., 4]
            Expected quaternion.
        atol : float
            Absolute tolerance.
        """
        res = self.space.quaternion_from_tait_bryan_angles(
            tait_bryan_angles, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_tait_bryan_angles_from_quaternion(
        self, quaternion, extrinsic, zyx, expected, atol=gs.atol
    ):
        """Check Tait-Bryan angles from quaternion.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]
            Quaternion.
        extrinsic : bool
            Whether the angles are extrinsic or intrinsic.
        zyx : bool
            Whether the angles are in zyx or xyz order.
        expected : array-like, shape=[..., 3]
            Expected Tait-Bryan angles.
        atol : float
            Absolute tolerance.
        """
        res = self.space.tait_bryan_angles_from_quaternion(
            quaternion, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_rotation_vector_from_tait_bryan_angles(
        self, tait_bryan_angles, extrinsic, zyx, expected, atol=gs.atol
    ):
        """Check rotation vector from Tait-Bryan angles.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]
            Tait-Bryan angles.
        extrinsic : bool
            Whether the angles are extrinsic or intrinsic.
        zyx : bool
            Whether the angles are in zyx or xyz order.
        expected : array-like, shape=[..., 3]
            Expected rotation vector.
        atol : float
            Absolute tolerance.
        """
        res = self.space.rotation_vector_from_tait_bryan_angles(
            tait_bryan_angles, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_tait_bryan_angles_from_rotation_vector(
        self, rot_vec, extrinsic, zyx, expected, atol=gs.atol
    ):
        """Check Tait-Bryan angles from rotation vector.

        Parameters
        ----------
        rot_vec : array-like, shape=[..., 3]
            Rotation vector.
        extrinsic : bool
            Whether the angles are extrinsic or intrinsic.
        zyx : bool
            Whether the angles are in zyx or xyz order.
        expected : array-like, shape=[..., 3]
            Expected Tait-Bryan angles.
        atol : float
            Absolute tolerance.
        """
        res = self.space.tait_bryan_angles_from_rotation_vector(
            rot_vec, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_left_jacobian_translation_det(self, point, expected, atol=gs.atol):
        """Check the determinant of the left translation Jacobian.

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point on the group.
        expected : array-like, shape=[...]
            Expected determinant.
        atol : float
            Absolute tolerance.
        """
        jacobian = self.space.jacobian_translation(point=point, left=True)
        result = gs.linalg.det(jacobian)
        self.assertAllClose(result, expected, atol=atol)

    def test_quaternion_from_matrix_after_matrix_from_quaternion(
        self, quaternion, atol=gs.atol
    ):
        """Check quaternion after rotation matrix gives back the quaternion.

        Quaternions are compared up to sign.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]
            Quaternion.
        atol : float
            Absolute tolerance.
        """
        mat = self.space.matrix_from_quaternion(quaternion)
        quaternion_ = self.space.quaternion_from_matrix(mat)
        self._assert_quaternion(quaternion_, quaternion, atol)

    def test_matrix_from_quaternion_after_quaternion_from_matrix(
        self, rot_mat, atol=gs.atol
    ):
        """Check rotation matrix after quaternion gives back the matrix.

        Parameters
        ----------
        rot_mat : array-like, shape=[..., 3, 3]
            Rotation matrix.
        atol : float
            Absolute tolerance.
        """
        quaternion = self.space.quaternion_from_matrix(rot_mat)
        rot_mat_ = self.space.matrix_from_quaternion(quaternion)
        self.assertAllClose(rot_mat_, rot_mat, atol=atol)

    def test_quaternion_from_rotation_vector_after_rotation_vector_from_quaternion(
        self, quaternion, atol=gs.atol
    ):
        """Check quaternion after rotation vector gives back the quaternion.

        Quaternions are compared up to sign.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]
            Quaternion.
        atol : float
            Absolute tolerance.
        """
        rot_vec = self.space.rotation_vector_from_quaternion(quaternion)
        quaternion_ = self.space.quaternion_from_rotation_vector(rot_vec)
        self._assert_quaternion(quaternion_, quaternion, atol)

    def test_rotation_vector_from_quaternion_after_quaternion_from_rotation_vector(
        self, rot_vec, atol=gs.atol
    ):
        """Check rotation vector after quaternion gives back the vector.

        Parameters
        ----------
        rot_vec : array-like, shape=[..., 3]
            Rotation vector.
        atol : float
            Absolute tolerance.
        """
        quaternion = self.space.quaternion_from_rotation_vector(rot_vec)
        rot_vec_ = self.space.rotation_vector_from_quaternion(quaternion)
        self.assertAllClose(rot_vec_, rot_vec, atol=atol)

    def test_tait_bryan_angles_from_matrix_after_matrix_from_tait_bryan_angles(
        self, tait_bryan_angles, extrinsic, zyx, atol=gs.atol
    ):
        """Check Tait-Bryan angles after rotation matrix give back the angles.

        Angles are compared through their rotation matrices if they differ.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]
            Tait-Bryan angles.
        extrinsic : bool
            Whether the angles are extrinsic or intrinsic.
        zyx : bool
            Whether the angles are in zyx or xyz order.
        atol : float
            Absolute tolerance.
        """
        mat = self.space.matrix_from_tait_bryan_angles(
            tait_bryan_angles, extrinsic=extrinsic, zyx=zyx
        )
        angles_ = self.space.tait_bryan_angles_from_matrix(
            mat, extrinsic=extrinsic, zyx=zyx
        )
        self._assert_tait_bryan_angles(
            angles_, tait_bryan_angles, extrinsic=extrinsic, zyx=zyx, atol=atol
        )

    def test_matrix_from_tait_bryan_angles_after_tait_bryan_angles_from_matrix(
        self, rot_mat, extrinsic, zyx, atol=gs.atol
    ):
        """Check rotation matrix after Tait-Bryan angles gives back the matrix.

        Parameters
        ----------
        rot_mat : array-like, shape=[..., 3, 3]
            Rotation matrix.
        extrinsic : bool
            Whether the angles are extrinsic or intrinsic.
        zyx : bool
            Whether the angles are in zyx or xyz order.
        atol : float
            Absolute tolerance.
        """
        angles = self.space.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic=extrinsic, zyx=zyx
        )
        rot_mat_ = self.space.matrix_from_tait_bryan_angles(
            angles, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(rot_mat_, rot_mat, atol=atol)

    def test_quaternion_from_tait_bryan_angles_after_tait_bryan_angles_from_quaternion(
        self, quaternion, extrinsic, zyx, atol=gs.atol
    ):
        """Check quaternion after Tait-Bryan angles gives back the quaternion.

        Quaternions are compared up to sign.

        Parameters
        ----------
        quaternion : array-like, shape=[..., 4]
            Quaternion.
        extrinsic : bool
            Whether the angles are extrinsic or intrinsic.
        zyx : bool
            Whether the angles are in zyx or xyz order.
        atol : float
            Absolute tolerance.
        """
        angles = self.space.tait_bryan_angles_from_quaternion(
            quaternion, extrinsic=extrinsic, zyx=zyx
        )
        quaternion_ = self.space.quaternion_from_tait_bryan_angles(
            angles, extrinsic=extrinsic, zyx=zyx
        )
        self._assert_quaternion(quaternion_, quaternion, atol=atol)

    def test_tait_bryan_angles_from_quaternion_after_quaternion_from_tait_bryan_angles(
        self, tait_bryan_angles, extrinsic, zyx, atol=gs.atol
    ):
        """Check Tait-Bryan angles after quaternion give back the angles.

        Angles are compared through their rotation matrices if they differ.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]
            Tait-Bryan angles.
        extrinsic : bool
            Whether the angles are extrinsic or intrinsic.
        zyx : bool
            Whether the angles are in zyx or xyz order.
        atol : float
            Absolute tolerance.
        """
        quaternion = self.space.quaternion_from_tait_bryan_angles(
            tait_bryan_angles, extrinsic=extrinsic, zyx=zyx
        )
        angles_ = self.space.tait_bryan_angles_from_quaternion(
            quaternion, extrinsic=extrinsic, zyx=zyx
        )
        self._assert_tait_bryan_angles(
            angles_, tait_bryan_angles, extrinsic=extrinsic, zyx=zyx, atol=atol
        )

    def test_tait_bryan_angles_from_rotation_vector_after_rotation_vector_from_tait_bryan_angles(
        self, tait_bryan_angles, extrinsic, zyx, atol=gs.atol
    ):
        """Check Tait-Bryan angles after rotation vector give back the angles.

        Angles are compared through their rotation matrices if they differ.

        Parameters
        ----------
        tait_bryan_angles : array-like, shape=[..., 3]
            Tait-Bryan angles.
        extrinsic : bool
            Whether the angles are extrinsic or intrinsic.
        zyx : bool
            Whether the angles are in zyx or xyz order.
        atol : float
            Absolute tolerance.
        """
        rot_vec = self.space.rotation_vector_from_tait_bryan_angles(
            tait_bryan_angles, extrinsic=extrinsic, zyx=zyx
        )
        angles_ = self.space.tait_bryan_angles_from_rotation_vector(
            rot_vec, extrinsic=extrinsic, zyx=zyx
        )
        self._assert_tait_bryan_angles(
            angles_, tait_bryan_angles, extrinsic=extrinsic, zyx=zyx, atol=atol
        )

    def test_rotation_vector_from_tait_bryan_angles_after_tait_bryan_angles_from_rotation_vector(
        self, rot_vec, extrinsic, zyx, atol=gs.atol
    ):
        """Check rotation vector after Tait-Bryan angles gives back the vector.

        Parameters
        ----------
        rot_vec : array-like, shape=[..., 3]
            Rotation vector.
        extrinsic : bool
            Whether the angles are extrinsic or intrinsic.
        zyx : bool
            Whether the angles are in zyx or xyz order.
        atol : float
            Absolute tolerance.
        """
        angles = self.space.tait_bryan_angles_from_rotation_vector(
            rot_vec, extrinsic=extrinsic, zyx=zyx
        )
        rot_vec_ = self.space.rotation_vector_from_tait_bryan_angles(
            angles, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(rot_vec_, rot_vec, atol=atol)

    def test_group_exp_after_log_with_angles_close_to_pi(self, point, base_point):
        """Check group exp after log for angles close to pi, up to sign.

        Parameters
        ----------
        point : array-like, shape=[3]
            Rotation vector.
        base_point : array-like, shape=[3]
            Rotation vector.
        """
        result = self.space.exp(self.space.log(point, base_point), base_point)
        expected = self.space.regularize(point)
        inv_expected = -expected

        self.assertTrue(
            gs.allclose(result, expected, atol=5e-3)
            or gs.allclose(result, inv_expected, atol=5e-3)
        )

    def test_group_log_after_exp_with_angles_close_to_pi(self, tangent_vec, base_point):
        """Check group log after exp for angles close to pi, up to sign.

        Parameters
        ----------
        tangent_vec : array-like, shape=[3]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[3]
            Rotation vector.
        """
        result = self.space.log(self.space.exp(tangent_vec, base_point), base_point)
        reg_tangent_vec = self.space.regularize_tangent_vec(
            tangent_vec=tangent_vec, base_point=base_point
        )
        expected = reg_tangent_vec
        inv_expected = -expected
        self.assertTrue(
            gs.allclose(result, expected, atol=5e-3)
            or gs.allclose(result, inv_expected, atol=5e-3)
        )

    def test_rotation_vector_and_rotation_matrix_with_angles_close_to_pi(self, point):
        """Check rotation vector after matrix for angles close to pi, up to sign.

        Parameters
        ----------
        point : array-like, shape=[3]
            Rotation vector.
        """
        mat = self.space.matrix_from_rotation_vector(point)
        result = self.space.rotation_vector_from_matrix(mat)
        expected1 = self.space.regularize(point)
        expected2 = -1 * expected1
        expected = gs.allclose(result, expected1) or gs.allclose(result, expected2)
        self.assertTrue(expected)

    def test_quaternion_and_matrix_with_angles_close_to_pi(self, point):
        """Check matrix after quaternion for angles close to pi, up to inversion.

        Parameters
        ----------
        point : array-like, shape=[3]
            Rotation vector.
        """
        mat = self.space.matrix_from_rotation_vector(point)
        quat = self.space.quaternion_from_matrix(mat)
        result = self.space.matrix_from_quaternion(quat)
        expected1 = mat
        expected2 = gs.linalg.inv(mat)
        expected = gs.allclose(result, expected1) or gs.allclose(result, expected2)
        self.assertTrue(expected)

    def test_quaternion_and_rotation_vector_with_angles_close_to_pi(self, point):
        """Check rotation vector after quaternion for angles close to pi.

        Rotation vectors are compared up to sign.

        Parameters
        ----------
        point : array-like, shape=[3]
            Rotation vector.
        """
        quaternion = self.space.quaternion_from_rotation_vector(point)
        result = self.space.rotation_vector_from_quaternion(quaternion)
        expected1 = self.space.regularize(point)
        expected2 = -1 * expected1
        expected = gs.allclose(result, expected1) or gs.allclose(result, expected2)
        self.assertTrue(expected)

    def test_compose_regularize_angles_close_to_pi(self, point):
        """Check composing with the identity regularizes, for angles close to pi.

        Parameters
        ----------
        point : array-like, shape=[3]
            Rotation vector.
        """
        result = self.space.compose(point, self.space.identity)
        expected = self.space.regularize(point)
        inv_expected = -expected
        self.assertTrue(
            gs.allclose(result, expected) or gs.allclose(result, inv_expected)
        )

        result = self.space.compose(self.space.identity, point)
        expected = self.space.regularize(point)
        inv_expected = -expected
        self.assertTrue(
            gs.allclose(result, expected) or gs.allclose(result, inv_expected)
        )
