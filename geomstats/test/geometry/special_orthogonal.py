import pytest

import geomstats.backend as gs
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.geometry.base import LieGroupTestCase, _ProjectionTestCaseMixins
from geomstats.test.random import get_random_quaternion
from geomstats.test.vectorization import generate_vectorization_data

# TODO: update _get to random


class _SpecialOrthogonalTestCaseMixins:
    def _get_vec(self, n_points=1):
        batch_shape = (n_points,) if n_points > 1 else ()
        return gs.random.normal(size=batch_shape + (self.space.dim,))

    def _get_skew_sym_matrix(self, n_points=1):
        return SkewSymmetricMatrices(self.space.n).random_point(n_points)

    def _get_rotation_matrix(self, n_points=1):
        return SpecialOrthogonal(n=self.space.n).random_point(n_points)

    def test_skew_matrix_from_vector(self, vec, expected, atol):
        mat = self.space.skew_matrix_from_vector(vec)
        self.assertAllClose(mat, expected)

    @pytest.mark.vec
    def test_skew_matrix_from_vector_vec(self, n_reps, atol):
        vec = self._get_vec()
        expected = self.space.skew_matrix_from_vector(vec)

        vec_data = generate_vectorization_data(
            data=[dict(vec=vec, expected=expected, atol=atol)],
            arg_names=["vec"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_vector_from_skew_matrix(self, mat, expected, atol):
        vec = self.space.vector_from_skew_matrix(mat)
        self.assertAllClose(vec, expected, atol=atol)

    @pytest.mark.vec
    def test_vector_from_skew_matrix_vec(self, n_reps, atol):
        mat = self._get_skew_sym_matrix()
        expected = self.space.vector_from_skew_matrix(mat)

        vec_data = generate_vectorization_data(
            data=[dict(mat=mat, expected=expected, atol=atol)],
            arg_names=["mat"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_vector_from_skew_matrix_after_skew_matrix_from_vector(
        self, n_points, atol
    ):
        vec = self._get_vec(n_points)
        mat = self.space.skew_matrix_from_vector(vec)
        vec_ = self.space.vector_from_skew_matrix(mat)
        self.assertAllClose(vec_, vec, atol=atol)

    @pytest.mark.random
    def test_skew_matrix_from_vector_after_vector_from_skew_matrix(
        self, n_points, atol
    ):
        mat = self._get_skew_sym_matrix(n_points)
        vec = self.space.vector_from_skew_matrix(mat)
        mat_ = self.space.skew_matrix_from_vector(vec)
        self.assertAllClose(mat_, mat, atol=atol)

    def test_rotation_vector_from_matrix(self, rot_mat, expected, atol):
        rot_vec = self.space.rotation_vector_from_matrix(rot_mat)
        self.assertAllClose(rot_vec, expected, atol=atol)

    @pytest.mark.vec
    def test_rotation_vector_from_matrix_vec(self, n_reps, atol):
        rot_mat = self._get_rotation_matrix()
        expected = self.space.rotation_vector_from_matrix(rot_mat)

        vec_data = generate_vectorization_data(
            data=[dict(rot_mat=rot_mat, expected=expected, atol=atol)],
            arg_names=["rot_mat"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_matrix_from_rotation_vector(self, rot_vec, expected, atol):
        rot_mat = self.space.matrix_from_rotation_vector(rot_vec)
        self.assertAllClose(rot_mat, expected, atol=atol)

    @pytest.mark.vec
    def test_matrix_from_rotation_vector_vec(self, n_reps, atol):
        rot_vec = self._get_vec()
        expected = self.space.matrix_from_rotation_vector(rot_vec)

        vec_data = generate_vectorization_data(
            data=[dict(rot_vec=rot_vec, expected=expected, atol=atol)],
            arg_names=["rot_vec"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_rotation_vector_from_matrix_after_matrix_from_rotation_vector(
        self, n_points, atol
    ):
        vec = self._get_vec(n_points)
        mat = self.space.matrix_from_rotation_vector(vec)
        vec_ = self.space.rotation_vector_from_matrix(mat)
        self.assertAllClose(vec, vec_, atol=atol)

    @pytest.mark.random
    def test_matrix_from_rotation_vector_after_rotation_vector_from_matrix(
        self, n_points, atol
    ):
        mat = self._get_rotation_matrix(n_points)
        vec = self.space.rotation_vector_from_matrix(mat)
        mat_ = self.space.matrix_from_rotation_vector(vec)
        self.assertAllClose(mat, mat_, atol=atol)


class SpecialOrthogonalVectorsTestCase(
    _ProjectionTestCaseMixins, _SpecialOrthogonalTestCaseMixins, LieGroupTestCase
):
    # TODO: add test on projection matrix belongs?

    def _get_point_to_project(self, n_points):
        batch_shape = (n_points,) if n_points > 1 else ()
        return gs.random.normal(size=batch_shape + self.space.shape)

    @pytest.mark.vec
    def test_projection_vec(self, n_reps, atol):
        # TODO: review class code design

        point = gs.random.normal(size=(self.space.n, self.space.n))
        proj_point = self.space.projection(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=proj_point, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)


class SpecialOrthogonal2VectorsTestCase(SpecialOrthogonalVectorsTestCase):
    pass


class SpecialOrthogonal3VectorsTestCase(SpecialOrthogonalVectorsTestCase):
    def _assert_quaternion(self, quaternion_, quaternion, atol):
        if quaternion_.ndim == 1:
            quaternion_ = gs.expand_dims(quaternion_, axis=0)

        if quaternion.ndim == 1:
            quaternion = gs.expand_dims(quaternion, axis=0)

        for quaternion_pt_, quaternion_pt in zip(quaternion_, quaternion):
            try:
                self.assertAllClose(quaternion_pt_, quaternion_pt, atol=atol)
            except AssertionError:
                self.assertAllClose(-quaternion_pt_, quaternion_pt, atol=atol)

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

    def _get_random_angles(self, n_points=1):
        size = (n_points, 3) if n_points > 1 else 3
        return gs.random.uniform(low=-gs.pi, high=gs.pi, size=size)

    def test_quaternion_from_matrix(self, rot_mat, expected, atol):
        res = self.space.quaternion_from_matrix(rot_mat)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_quaternion_from_matrix_vec(self, n_reps, atol):
        rot_mat = self._get_rotation_matrix()
        expected = self.space.quaternion_from_matrix(rot_mat)

        vec_data = generate_vectorization_data(
            data=[dict(rot_mat=rot_mat, expected=expected, atol=atol)],
            arg_names=["rot_mat"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_matrix_from_quaternion(self, quaternion, expected, atol):
        res = self.space.matrix_from_quaternion(quaternion)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_matrix_from_quaternion_vec(self, n_reps, atol):
        quaternion = get_random_quaternion()
        expected = self.space.matrix_from_quaternion(quaternion)

        vec_data = generate_vectorization_data(
            data=[dict(quaternion=quaternion, expected=expected, atol=atol)],
            arg_names=["quaternion"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_quaternion_from_matrix_after_matrix_from_quaternion(self, n_points, atol):
        quaternion = get_random_quaternion(n_points)
        mat = self.space.matrix_from_quaternion(quaternion)
        quaternion_ = self.space.quaternion_from_matrix(mat)

        self._assert_quaternion(quaternion_, quaternion, atol)

    @pytest.mark.random
    def test_matrix_from_quaternion_after_quaternion_from_matrix(self, n_points, atol):
        mat = self._get_rotation_matrix(n_points)
        quaternion = self.space.quaternion_from_matrix(mat)
        mat_ = self.space.matrix_from_quaternion(quaternion)
        self.assertAllClose(mat_, mat, atol=atol)

    def test_quaternion_from_rotation_vector(self, rot_vec, expected, atol):
        res = self.space.quaternion_from_rotation_vector(rot_vec)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_quaternion_from_rotation_vector_vec(self, n_reps, atol):
        rot_vec = self._get_vec()
        expected = self.space.quaternion_from_rotation_vector(rot_vec)

        vec_data = generate_vectorization_data(
            data=[dict(rot_vec=rot_vec, expected=expected, atol=atol)],
            arg_names=["rot_vec"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_rotation_vector_from_quaternion(self, quaternion, expected, atol):
        res = self.space.rotation_vector_from_quaternion(quaternion)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_rotation_vector_from_quaternion_vec(self, n_reps, atol):
        quaternion = get_random_quaternion()
        expected = self.space.rotation_vector_from_quaternion(quaternion)

        vec_data = generate_vectorization_data(
            data=[dict(quaternion=quaternion, expected=expected, atol=atol)],
            arg_names=["quaternion"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_quaternion_from_rotation_vector_after_rotation_vector_from_quaternion(
        self, n_points, atol
    ):
        quaternion = get_random_quaternion(n_points)
        rot_vec = self.space.rotation_vector_from_quaternion(quaternion)
        quaternion_ = self.space.quaternion_from_rotation_vector(rot_vec)

        self._assert_quaternion(quaternion_, quaternion, atol)

    @pytest.mark.random
    def test_rotation_vector_from_quaternion_after_quaternion_from_rotation_vector(
        self, n_points, atol
    ):
        rot_vec = get_random_quaternion(n_points)
        quaternion = self.space.quaternion_from_rotation_vector(rot_vec)
        rot_vec_ = self.space.rotation_vector_from_quaternion(quaternion)
        self.assertAllClose(rot_vec_, rot_vec, atol=atol)

    def test_matrix_from_tait_bryan_angles(
        self, tait_bryan_angles, extrinsic, zyx, expected, atol
    ):

        res = self.space.matrix_from_tait_bryan_angles(
            tait_bryan_angles, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_matrix_from_tait_bryan_angles_vec(self, n_reps, extrinsic, zyx, atol):
        angles = self._get_random_angles()
        expected = self.space.matrix_from_tait_bryan_angles(
            angles, extrinsic=extrinsic, zyx=zyx
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tait_bryan_angles=angles,
                    extrinsic=extrinsic,
                    zyx=zyx,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tait_bryan_angles"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_tait_bryan_angles_from_matrix(
        self, rot_mat, extrinsic, zyx, expected, atol
    ):
        res = self.space.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_tait_bryan_angles_from_matrix_vec(self, n_reps, extrinsic, zyx, atol):
        rot_mat = self._get_rotation_matrix()
        expected = self.space.tait_bryan_angles_from_matrix(
            rot_mat, extrinsic=extrinsic, zyx=zyx
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    rot_mat=rot_mat,
                    extrinsic=extrinsic,
                    zyx=zyx,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["rot_mat"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_tait_bryan_angles_from_matrix_after_matrix_from_tait_bryan_angles(
        self, n_points, extrinsic, zyx, atol
    ):
        angles = self._get_random_angles(n_points)
        mat = self.space.matrix_from_tait_bryan_angles(
            angles, extrinsic=extrinsic, zyx=zyx
        )
        angles_ = self.space.tait_bryan_angles_from_matrix(
            mat, extrinsic=extrinsic, zyx=zyx
        )
        # self.assertAllClose(angles_, angles, atol=atol)
        self._assert_tait_bryan_angles(
            angles_, angles, extrinsic=extrinsic, zyx=zyx, atol=atol
        )

    @pytest.mark.random
    def test_matrix_from_tait_bryan_angles_after_tait_bryan_angles_from_matrix(
        self, n_points, extrinsic, zyx, atol
    ):
        mat = self._get_rotation_matrix(n_points)
        angles = self.space.tait_bryan_angles_from_matrix(
            mat, extrinsic=extrinsic, zyx=zyx
        )
        mat_ = self.space.matrix_from_tait_bryan_angles(
            angles, extrinsic=extrinsic, zyx=zyx
        )

        self.assertAllClose(mat_, mat, atol=atol)

    def test_quaternion_from_tait_bryan_angles(
        self, tait_bryan_angles, extrinsic, zyx, expected, atol
    ):
        res = self.space.quaternion_from_tait_bryan_angles(
            tait_bryan_angles, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_quaternion_from_tait_bryan_angles_vec(self, n_reps, extrinsic, zyx, atol):
        angles = self._get_random_angles()
        expected = self.space.quaternion_from_tait_bryan_angles(
            angles, extrinsic=extrinsic, zyx=zyx
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tait_bryan_angles=angles,
                    extrinsic=extrinsic,
                    zyx=zyx,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tait_bryan_angles"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_tait_bryan_angles_from_quaternion(
        self, quaternion, extrinsic, zyx, expected, atol
    ):
        res = self.space.tait_bryan_angles_from_quaternion(
            quaternion, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_tait_bryan_angles_from_quaternion_vec(self, n_reps, extrinsic, zyx, atol):
        quaternion = get_random_quaternion()
        expected = self.space.tait_bryan_angles_from_quaternion(
            quaternion, extrinsic=extrinsic, zyx=zyx
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    quaternion=quaternion,
                    extrinsic=extrinsic,
                    zyx=zyx,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["quaternion"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_quaternion_from_tait_bryan_angles_after_tait_bryan_angles_from_quaternion(
        self, n_points, extrinsic, zyx, atol
    ):
        quaternion = get_random_quaternion(n_points)
        angles = self.space.tait_bryan_angles_from_quaternion(
            quaternion, extrinsic=extrinsic, zyx=zyx
        )
        quaternion_ = self.space.quaternion_from_tait_bryan_angles(
            angles, extrinsic=extrinsic, zyx=zyx
        )
        self._assert_quaternion(quaternion_, quaternion, atol=atol)

    @pytest.mark.random
    def test_tait_bryan_angles_from_quaternion_after_quaternion_from_tait_bryan_angles(
        self, n_points, extrinsic, zyx, atol
    ):
        angles = self._get_random_angles(n_points)
        quaternion = self.space.quaternion_from_tait_bryan_angles(
            angles, extrinsic=extrinsic, zyx=zyx
        )
        angles_ = self.space.tait_bryan_angles_from_quaternion(
            quaternion, extrinsic=extrinsic, zyx=zyx
        )

        self._assert_tait_bryan_angles(
            angles_, angles, extrinsic=extrinsic, zyx=zyx, atol=atol
        )

    def test_rotation_vector_from_tait_bryan_angles(
        self, tait_bryan_angles, extrinsic, zyx, expected, atol
    ):
        res = self.space.rotation_vector_from_tait_bryan_angles(
            tait_bryan_angles, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_rotation_vector_from_tait_bryan_angles_vec(
        self, n_reps, extrinsic, zyx, atol
    ):
        angles = self._get_random_angles()
        expected = self.space.rotation_vector_from_tait_bryan_angles(
            angles, extrinsic=extrinsic, zyx=zyx
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tait_bryan_angles=angles,
                    extrinsic=extrinsic,
                    zyx=zyx,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tait_bryan_angles"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_tait_bryan_angles_from_rotation_vector(
        self, rot_vec, extrinsic, zyx, expected, atol
    ):
        res = self.space.tait_bryan_angles_from_rotation_vector(
            rot_vec, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_tait_bryan_angles_from_rotation_vector_vec(
        self, n_reps, extrinsic, zyx, atol
    ):
        rot_vec = self._get_vec()
        expected = self.space.tait_bryan_angles_from_rotation_vector(
            rot_vec, extrinsic=extrinsic, zyx=zyx
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    rot_vec=rot_vec,
                    extrinsic=extrinsic,
                    zyx=zyx,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["rot_vec"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_tait_bryan_angles_from_rotation_vector_after_rotation_vector_from_tait_bryan_angles(
        self, n_points, extrinsic, zyx, atol
    ):
        angles = self._get_random_angles(n_points)
        rot_vec = self.space.rotation_vector_from_tait_bryan_angles(
            angles, extrinsic=extrinsic, zyx=zyx
        )
        angles_ = self.space.tait_bryan_angles_from_rotation_vector(
            rot_vec, extrinsic=extrinsic, zyx=zyx
        )

        self._assert_tait_bryan_angles(
            angles_, angles, extrinsic=extrinsic, zyx=zyx, atol=atol
        )

    @pytest.mark.random
    def test_rotation_vector_from_tait_bryan_angles_after_tait_bryan_angles_from_rotation_vector(
        self, n_points, extrinsic, zyx, atol
    ):
        rot_vec = self._get_vec(n_points)
        angles = self.space.tait_bryan_angles_from_rotation_vector(
            rot_vec, extrinsic=extrinsic, zyx=zyx
        )
        rot_vec_ = self.space.rotation_vector_from_tait_bryan_angles(
            angles, extrinsic=extrinsic, zyx=zyx
        )

        self.assertAllClose(rot_vec_, rot_vec, atol=atol)
