import pytest

import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import (
    SpecialOrthogonal,
    _SpecialOrthogonal2Vectors,
    _SpecialOrthogonal3Vectors,
)
from geomstats.test.random import (
    LieGroupVectorRandomDataGenerator,
    get_random_quaternion,
)
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.lie_group import (
    LieGroupTestCase,
    MatrixLieGroupTestCase,
)
from geomstats.test_cases.geometry.mixins import ProjectionTestCaseMixins

# TODO: improve random generation


class _SpecialOrthogonalTestCaseMixins:
    def _get_random_rotation_vector(self, n_points=1):
        if self.space.n == 2:
            return _SpecialOrthogonal2Vectors(equip=False).random_point(n_points)

        if self.space.n == 3:
            return _SpecialOrthogonal3Vectors(equip=False).random_point(n_points)

        raise NotImplementedError(
            f"Unable to create random orthogonal vector for `n={self.space.n}`"
        )

    def _get_random_rotation_matrix(self, n_points=1):
        return SpecialOrthogonal(n=self.space.n, equip=False).random_point(n_points)

    def test_rotation_vector_from_matrix(self, rot_mat, expected, atol):
        rot_vec = self.space.rotation_vector_from_matrix(rot_mat)
        self.assertAllClose(rot_vec, expected, atol=atol)

    @pytest.mark.vec
    def test_rotation_vector_from_matrix_vec(self, n_reps, atol):
        rot_mat = self._get_random_rotation_matrix()

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
        rot_vec = self._get_random_rotation_vector()
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
        vec = self._get_random_rotation_vector(n_points)
        mat = self.space.matrix_from_rotation_vector(vec)
        vec_ = self.space.rotation_vector_from_matrix(mat)
        self.assertAllClose(vec, vec_, atol=atol)

    @pytest.mark.random
    def test_matrix_from_rotation_vector_after_rotation_vector_from_matrix(
        self, n_points, atol
    ):
        mat = self._get_random_rotation_matrix(n_points)
        vec = self.space.rotation_vector_from_matrix(mat)
        mat_ = self.space.matrix_from_rotation_vector(vec)
        self.assertAllClose(mat, mat_, atol=atol)


class SpecialOrthogonalMatricesTestCase(
    _SpecialOrthogonalTestCaseMixins, MatrixLieGroupTestCase, LevelSetTestCase
):
    def test_are_antipodals(self, rotation_mat1, rotation_mat2, expected, atol):
        res = self.space.are_antipodals(rotation_mat1, rotation_mat2)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_are_antipodals_vec(self, n_reps, atol):
        rotation_mat1 = self._get_random_rotation_matrix()
        rotation_mat2 = self._get_random_rotation_matrix()

        expected = self.space.are_antipodals(rotation_mat1, rotation_mat2)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    rotation_mat1=rotation_mat1,
                    rotation_mat2=rotation_mat2,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["rotation_mat1", "rotation_mat2"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)


class SpecialOrthogonalVectorsTestCase(
    ProjectionTestCaseMixins, _SpecialOrthogonalTestCaseMixins, LieGroupTestCase
):
    # TODO: add test on projection matrix belongs?
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = LieGroupVectorRandomDataGenerator(self.space)

    def _get_random_rotation_vector(self, n_points=1):
        return self.space.random_point(n_points)

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

    def _get_random_angles(self, n_points=1):
        size = (n_points, 3) if n_points > 1 else 3
        return gs.random.uniform(low=-gs.pi, high=gs.pi, size=size)

    def test_group_exp_after_log_with_angles_close_to_pi(self, point, base_point):
        result = self.space.exp(self.space.log(point, base_point), base_point)
        expected = self.space.regularize(point)
        inv_expected = -expected

        self.assertTrue(
            gs.allclose(result, expected, atol=5e-3)
            or gs.allclose(result, inv_expected, atol=5e-3)
        )

    def test_group_log_after_exp_with_angles_close_to_pi(self, tangent_vec, base_point):
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
        mat = self.space.matrix_from_rotation_vector(point)
        result = self.space.rotation_vector_from_matrix(mat)
        expected1 = self.space.regularize(point)
        expected2 = -1 * expected1
        expected = gs.allclose(result, expected1) or gs.allclose(result, expected2)
        self.assertTrue(expected)

    def test_quaternion_from_matrix(self, rot_mat, expected, atol):
        res = self.space.quaternion_from_matrix(rot_mat)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_quaternion_from_matrix_vec(self, n_reps, atol):
        rot_mat = self._get_random_rotation_matrix()
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
        mat = self._get_random_rotation_matrix(n_points)
        quaternion = self.space.quaternion_from_matrix(mat)
        mat_ = self.space.matrix_from_quaternion(quaternion)
        self.assertAllClose(mat_, mat, atol=atol)

    def test_quaternion_and_matrix_with_angles_close_to_pi(self, point):
        mat = self.space.matrix_from_rotation_vector(point)
        quat = self.space.quaternion_from_matrix(mat)
        result = self.space.matrix_from_quaternion(quat)
        expected1 = mat
        expected2 = gs.linalg.inv(mat)
        expected = gs.allclose(result, expected1) or gs.allclose(result, expected2)
        self.assertTrue(expected)

    def test_quaternion_from_rotation_vector(self, rot_vec, expected, atol):
        res = self.space.quaternion_from_rotation_vector(rot_vec)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_quaternion_from_rotation_vector_vec(self, n_reps, atol):
        rot_vec = self._get_random_rotation_vector()
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
        rot_vec = self._get_random_rotation_vector(n_points)
        quaternion = self.space.quaternion_from_rotation_vector(rot_vec)
        rot_vec_ = self.space.rotation_vector_from_quaternion(quaternion)
        self.assertAllClose(rot_vec_, rot_vec, atol=atol)

    def test_quaternion_and_rotation_vector_with_angles_close_to_pi(self, point):
        quaternion = self.space.quaternion_from_rotation_vector(point)
        result = self.space.rotation_vector_from_quaternion(quaternion)
        expected1 = self.space.regularize(point)
        expected2 = -1 * expected1
        expected = gs.allclose(result, expected1) or gs.allclose(result, expected2)
        self.assertTrue(expected)

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
        rot_mat = self._get_random_rotation_matrix()
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
        mat = self._get_random_rotation_matrix(n_points)
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
        rot_vec = self._get_random_rotation_vector()
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
        rot_vec = self._get_random_rotation_vector(n_points)
        angles = self.space.tait_bryan_angles_from_rotation_vector(
            rot_vec, extrinsic=extrinsic, zyx=zyx
        )
        rot_vec_ = self.space.rotation_vector_from_tait_bryan_angles(
            angles, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(rot_vec_, rot_vec, atol=atol)

    def test_left_jacobian_translation_det(self, point, expected, atol):
        jacobian = self.space.jacobian_translation(point=point, left=True)
        result = gs.linalg.det(jacobian)
        self.assertAllClose(result, expected, atol=atol)

    def test_compose_regularize_angles_close_to_pi(self, point):
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
