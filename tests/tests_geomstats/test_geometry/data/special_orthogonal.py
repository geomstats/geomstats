import itertools
import math

import pytest
from polpo.testing.data import CaseData

import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

from .base import LevelSetTestData
from .lie_group import LieGroupTestData, MatrixLieGroupTestData
from .mixins import ProjectionMixinsTestData


def sample_matrix(theta, mul=1.0):
    return gs.array(
        [[gs.cos(theta), mul * gs.sin(theta)], [gs.sin(theta), gs.cos(theta)]]
    )


class _SpecialOrthogonalMixinsTestData:
    def rotation_vector_from_matrix_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="rot_mat",
            samplers={"rot_mat": "random_rotation_matrix"},
        )

    def matrix_from_rotation_vector_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="rot_vec",
            samplers={"rot_vec": "random_rotation_vector"},
        )

    def rotation_vector_from_matrix_after_matrix_from_rotation_vector_test_data(self):
        return self.generate_random_data(
            arg_names="rot_vec",
            samplers={"rot_vec": "random_rotation_vector"},
        )

    def matrix_from_rotation_vector_after_rotation_vector_from_matrix_test_data(self):
        return self.generate_random_data(
            arg_names="rot_mat",
            samplers={"rot_mat": "random_rotation_matrix"},
        )


class SpecialOrthogonalMatricesTestData(
    _SpecialOrthogonalMixinsTestData, MatrixLieGroupTestData, LevelSetTestData
):
    def are_antipodals_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("rotation_mat1", "rotation_mat2"),
            samplers={
                "rotation_mat1": "random_rotation_matrix",
                "rotation_mat2": "random_rotation_matrix",
            },
        )


class SpecialOrthogonalMatrices2TestData(CaseData):
    def belongs_test_data(self):
        theta = gs.pi / 3
        data = [
            dict(point=sample_matrix(theta, mul=-1.0), expected=True),
            dict(point=sample_matrix(theta, mul=1.0), expected=False),
            dict(point=gs.zeros((2, 3)), expected=False),
            dict(point=gs.zeros((2, 2, 3)), expected=gs.array([False, False])),
            dict(
                point=gs.stack(
                    [
                        sample_matrix(theta / 2, mul=-1.0),
                        sample_matrix(theta / 2, mul=1.0),
                    ]
                ),
                expected=gs.array([True, False]),
            ),
        ]
        return data

    def identity_test_data(self):
        data = [
            dict(expected=gs.eye(2)),
        ]
        return data

    def is_tangent_test_data(self):
        theta = gs.pi / 3
        vec_1 = gs.array([[0.0, -theta], [theta, 0.0]])
        vec_2 = gs.array([[0.0, -theta], [theta, 1.0]])
        data = [
            dict(vector=vec_1, base_point=None, expected=True),
            dict(vector=vec_2, base_point=None, expected=False),
        ]
        return data

    def to_tangent_test_data(self):
        theta = 1.0
        data = [
            dict(
                vector=gs.array([[0.0, -theta], [theta, 0.0]]),
                base_point=None,
                expected=gs.array([[0.0, -theta], [theta, 0.0]]),
            ),
            dict(
                vector=gs.array([[1.0, -math.pi], [math.pi, 1.0]]),
                base_point=gs.array(
                    [
                        [gs.cos(math.pi), -1 * gs.sin(math.pi)],
                        [gs.sin(math.pi), gs.cos(math.pi)],
                    ]
                ),
                expected=gs.array([[0.0, -math.pi], [math.pi, 0.0]]),
            ),
        ]
        return data

    def matrix_from_rotation_vector_test_data(self):
        data = [
            dict(
                rot_vec=gs.array([gs.pi / 3]),
                expected=gs.array(
                    [[1.0 / 2, -gs.sqrt(3.0) / 2], [gs.sqrt(3.0) / 2, 1.0 / 2]]
                ),
            ),
        ]
        return data


class SpecialOrthogonalMatrices3TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.zeros((2, 3)), expected=False),
        ]
        return data

    def identity_test_data(self):
        data = [
            dict(expected=gs.eye(3)),
        ]
        return data

    def are_antipodals_test_data(self):
        mat1 = gs.eye(3)
        mat2 = gs.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

        rot = SpecialOrthogonal(
            3, point_type="vector", equip=False
        ).matrix_from_rotation_vector
        base_mat = rot(gs.array([-1.5, -1.0, -1.0]))
        nearby_mat = gs.matmul(base_mat, rot(gs.array([1e-6, 0.0, 0.0])))
        antipodal_mat = gs.matmul(base_mat, rot(gs.pi * gs.array([0.0, 0.6, 0.8])))
        near_antipodal_mat = gs.matmul(
            base_mat, rot((gs.pi - 1e-3) * gs.array([0.0, 0.6, 0.8]))
        )

        data = [
            dict(rotation_mat1=mat1, rotation_mat2=mat2, expected=gs.array(True)),
            dict(
                rotation_mat1=gs.stack([mat1, mat2]),
                rotation_mat2=gs.stack([mat2, mat2]),
                expected=gs.array([True, False]),
            ),
            dict(
                rotation_mat1=base_mat,
                rotation_mat2=nearby_mat,
                expected=gs.array(False),
            ),
            dict(
                rotation_mat1=base_mat,
                rotation_mat2=antipodal_mat,
                expected=gs.array(True),
            ),
            dict(
                rotation_mat1=base_mat,
                rotation_mat2=near_antipodal_mat,
                expected=gs.array(False),
            ),
        ]
        return data

    def log_test_data(self):
        rot = SpecialOrthogonal(
            3, point_type="vector", equip=False
        ).matrix_from_rotation_vector
        base_mat = rot(gs.array([-1.5, -1.0, -1.0]))
        nearby_mat = gs.matmul(base_mat, rot(gs.array([1e-6, 0.0, 0.0])))
        skew = gs.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1e-6], [0.0, 1e-6, 0.0]])

        data = [
            dict(
                point=nearby_mat,
                base_point=base_mat,
                expected=gs.matmul(base_mat, skew),
            ),
        ]
        return data

    def matrix_from_rotation_vector_test_data(self):
        rot_vec_3 = 1e-11 * gs.array([12.0, 1.0, -81.0])
        angle = gs.linalg.norm(rot_vec_3)
        skew_rot_vec_3 = 1e-11 * gs.array(
            [[0.0, 81.0, 1.0], [-81.0, 0.0, -12.0], [-1.0, 12.0, 0.0]]
        )
        coef_1 = gs.sin(angle) / angle
        coef_2 = (1.0 - gs.cos(angle)) / (angle**2)
        expected_3 = (
            gs.eye(3)
            + coef_1 * skew_rot_vec_3
            + coef_2 * gs.matmul(skew_rot_vec_3, skew_rot_vec_3)
        )

        rot_vec_4 = gs.array([0.1, 1.3, -0.5])
        angle = gs.linalg.norm(rot_vec_4)
        skew_rot_vec_4 = gs.array(
            [[0.0, 0.5, 1.3], [-0.5, 0.0, -0.1], [-1.3, 0.1, 0.0]]
        )

        coef_1 = gs.sin(angle) / angle
        coef_2 = (1 - gs.cos(angle)) / (angle**2)
        expected_4 = (
            gs.eye(3)
            + coef_1 * skew_rot_vec_4
            + coef_2 * gs.matmul(skew_rot_vec_4, skew_rot_vec_4)
        )
        data = [
            dict(rot_vec=gs.array([0.0, 0.0, 0.0]), expected=gs.eye(3)),
            dict(
                rot_vec=gs.array([gs.pi / 3.0, 0.0, 0.0]),
                expected=gs.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 0.5, -gs.sqrt(3.0) / 2],
                        [0.0, gs.sqrt(3.0) / 2, 0.5],
                    ]
                ),
            ),
            dict(rot_vec=rot_vec_3, expected=expected_3),
            dict(rot_vec=rot_vec_4, expected=expected_4),
        ]
        return data


class SpecialOrthogonalVectorsTestData(
    ProjectionMixinsTestData, _SpecialOrthogonalMixinsTestData, LieGroupTestData
):
    pass


class SpecialOrthogonal2VectorsTestData(SpecialOrthogonalVectorsTestData):
    @pytest.mark.smoke
    def identity_test_data(self):
        data = [
            dict(expected=gs.zeros(1)),
        ]
        return data

    @pytest.mark.smoke
    def rotation_vector_from_matrix_test_data(self):
        angle = 0.12
        data = [
            dict(
                rot_mat=gs.array(
                    [
                        [gs.cos(angle), -gs.sin(angle)],
                        [gs.sin(angle), gs.cos(angle)],
                    ]
                ),
                expected=gs.array([0.12]),
            ),
        ]
        return data

    @pytest.mark.smoke
    def projection_test_data(self):
        data = [
            dict(
                point=gs.eye(2) + 1e-12 * gs.ones((2, 2)),
                expected=gs.array([0.0]),
            )
        ]
        return data

    @pytest.mark.smoke
    def compose_test_data(self):
        data = [
            dict(
                point_a=gs.array([0.12]),
                point_b=gs.array([-0.15]),
                expected=gs.array([-0.03]),
            )
        ]
        return data

    @pytest.mark.smoke
    def log_test_data(self):
        data = [
            dict(
                point=gs.array([2 * gs.pi / 5]),
                base_point=gs.array([gs.pi / 5]),
                expected=gs.array([1 * gs.pi / 5]),
            )
        ]
        return data

    @pytest.mark.smoke
    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([2 * gs.pi / 5]),
                base_point=gs.array([gs.pi / 5]),
                expected=gs.array([3 * gs.pi / 5]),
            )
        ]
        return data

    @pytest.mark.smoke
    def regularize_test_data(self):
        data = [
            dict(
                point=gs.array([2 * gs.pi + 1]),
                expected=gs.array([1.0]),
            )
        ]
        return data


class SpecialOrthogonal3VectorsTestData(SpecialOrthogonalVectorsTestData):
    angle_0 = gs.zeros(3)
    angle_close_0 = 1e-10 * gs.array([1.0, -1.0, 1.0])
    angle_close_pi_low = (gs.pi - 1e-9) / gs.sqrt(2.0) * gs.array([0.0, 1.0, -1.0])
    angle_pi = gs.pi * gs.array([1.0, 0, 0])
    angle_close_pi_high = (gs.pi + 1e-9) / gs.sqrt(3.0) * gs.array([-1.0, 1.0, -1])
    angle_in_pi_2pi = (gs.pi + 0.3) / gs.sqrt(5.0) * gs.array([-2.0, 1.0, 0.0])
    angle_close_2pi_low = (2.0 * gs.pi - 1e-9) / gs.sqrt(6.0) * gs.array([2.0, 1.0, -1])
    angle_2pi = 2.0 * gs.pi / gs.sqrt(3.0) * gs.array([1.0, 1.0, -1.0])
    angle_close_2pi_high = (
        (2.0 * gs.pi + 1e-9) / gs.sqrt(2.0) * gs.array([1.0, 0.0, -1.0])
    )

    elements = {
        "angle_0": angle_0,
        "angle_close_0": angle_close_0,
        "angle_close_pi_low": angle_close_pi_low,
        "angle_pi": angle_pi,
        "angle_close_pi_high": angle_close_pi_high,
        "angle_in_pi_2pi": angle_in_pi_2pi,
        "angle_close_2pi_low": angle_close_2pi_low,
        "angle_2pi": angle_2pi,
        "angle_close_2pi_high": angle_close_2pi_high,
    }

    angle_pi_6 = gs.pi / 6.0
    cos_angle_pi_6 = gs.cos(angle_pi_6)
    sin_angle_pi_6 = gs.sin(angle_pi_6)

    cos_angle_pi_12 = gs.cos(angle_pi_6 / 2)
    sin_angle_pi_12 = gs.sin(angle_pi_6 / 2)

    angles_close_to_pi = [
        "angle_close_pi_low",
        "angle_pi",
        "angle_close_pi_high",
    ]

    @pytest.mark.smoke
    def rotation_vector_from_matrix_test_data(self):
        angle = 0.12
        data = [
            dict(
                rot_mat=gs.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, gs.cos(angle), -gs.sin(angle)],
                        [0, gs.sin(angle), gs.cos(angle)],
                    ]
                ),
                expected=0.12 * gs.array([1.0, 0.0, 0.0]),
            ),
        ]
        return data

    @pytest.mark.smoke
    def projection_test_data(self):
        data = [
            dict(
                point=gs.eye(3) + 1e-12 * gs.ones((3, 3)),
                expected=gs.zeros(3),
            )
        ]
        return data

    @pytest.mark.smoke
    def tait_bryan_angles_from_matrix_test_data(self):
        cos_angle_pi_6 = self.cos_angle_pi_6
        sin_angle_pi_6 = self.sin_angle_pi_6
        angle_pi_6 = self.angle_pi_6
        xyz = gs.array(
            [
                [
                    [cos_angle_pi_6, -sin_angle_pi_6, 0.0],
                    [sin_angle_pi_6, cos_angle_pi_6, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [cos_angle_pi_6, 0.0, sin_angle_pi_6],
                    [0.0, 1.0, 0.0],
                    [-sin_angle_pi_6, 0.0, cos_angle_pi_6],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [0.0, cos_angle_pi_6, -sin_angle_pi_6],
                    [0.0, sin_angle_pi_6, cos_angle_pi_6],
                ],
            ]
        )
        zyx = gs.flip(xyz, axis=0)
        mat_data = {"xyz": xyz, "zyx": zyx}
        data = []

        for extrinsic, order in itertools.product([False, True], list(mat_data.keys())):
            for i in range(3):
                vec = gs.squeeze(gs.array_from_sparse([(0, i)], [angle_pi_6], (1, 3)))
                zyx = order == "zyx"
                data += [
                    dict(
                        rot_mat=mat_data[order][i],
                        extrinsic=extrinsic,
                        zyx=zyx,
                        expected=vec,
                    )
                ]
            data += [
                dict(
                    rot_mat=gs.eye(3),
                    extrinsic=extrinsic,
                    zyx=zyx,
                    expected=gs.zeros(3),
                )
            ]
        return data

    @pytest.mark.smoke
    def tait_bryan_angles_from_quaternion_test_data(self):
        cos_angle_pi_12 = self.cos_angle_pi_12
        sin_angle_pi_12 = self.sin_angle_pi_12
        angle_pi_6 = self.angle_pi_6

        xyz = gs.array(
            [
                [cos_angle_pi_12, 0.0, 0.0, sin_angle_pi_12],
                [cos_angle_pi_12, 0.0, sin_angle_pi_12, 0.0],
                [cos_angle_pi_12, sin_angle_pi_12, 0.0, 0.0],
            ]
        )

        zyx = gs.flip(xyz, axis=0)
        mat_data = {"xyz": xyz, "zyx": zyx}
        data = []
        e1 = gs.array([1.0, 0.0, 0.0, 0.0])
        for extrinsic, order in itertools.product([False, True], list(mat_data.keys())):
            for i in range(3):
                vec = gs.squeeze(gs.array_from_sparse([(0, i)], [angle_pi_6], (1, 3)))
                zyx = order == "zyx"
                data += [
                    dict(
                        quaternion=mat_data[order][i],
                        extrinsic=extrinsic,
                        zyx=zyx,
                        expected=vec,
                    )
                ]
            data += [
                dict(quaternion=e1, extrinsic=extrinsic, zyx=zyx, expected=gs.zeros(3))
            ]
        return data

    @pytest.mark.smoke
    def lie_bracket_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([0.0, 0.0, -1.0]),
                tangent_vec_b=gs.array([0.0, 0.0, -1.0]),
                base_point=None,
                expected=gs.zeros(3),
            ),
            dict(
                tangent_vec_a=gs.array([0.0, 0.0, 1.0]),
                tangent_vec_b=gs.array([0.0, 1.0, 0.0]),
                base_point=None,
                expected=gs.array([-1.0, 0.0, 0.0]),
            ),
        ]
        return data

    @pytest.mark.smoke
    def left_jacobian_translation_det_test_data(self):
        space = SpecialOrthogonal(3, point_type="vector")
        elements = self.elements
        smoke_data = []
        for angle_type in elements:
            point = elements[angle_type]
            angle = gs.linalg.norm(space.regularize(point))
            if angle_type in [
                "angle_0",
                "angle_close_0",
                "angle_2pi",
                "angle_close_2pi_high",
            ]:
                expected = 1.0 + angle**2 / 12.0 + angle**4 / 240.0
            else:
                expected = angle**2 / (4 * gs.sin(angle / 2) ** 2)
            smoke_data += [dict(point=point, expected=expected)]

        return smoke_data

    def _regularize_extreme_data(self):
        smoke_data = []
        for angle_type in [
            "angle_close_0",
            "angle_close_pi_low",
            "angle_pi",
            "angle_0",
        ]:
            smoke_data += [
                dict(
                    point=self.elements[angle_type],
                    expected=self.elements[angle_type],
                )
            ]
        point = self.elements["angle_close_pi_high"]
        norm = gs.linalg.norm(point)
        smoke_data += [dict(point=point, expected=point / norm * (norm - 2 * gs.pi))]

        for angle_type in ["angle_in_pi_2pi", "angle_close_2pi_low"]:
            point = self.elements[angle_type]
            angle = gs.linalg.norm(point)
            new_angle = gs.pi - (angle - gs.pi)

            point_initial = point
            expected = -(new_angle / angle) * point_initial
            smoke_data += [dict(point=point, expected=expected)]

        smoke_data += [
            dict(
                point=self.elements["angle_2pi"],
                expected=gs.array([0.0, 0.0, 0.0]),
            )
        ]
        point = self.elements["angle_close_2pi_high"]
        angle = gs.linalg.norm(point)
        new_angle = angle - 2 * gs.pi
        expected = new_angle * point / angle
        smoke_data += [dict(point=point, expected=expected)]
        return smoke_data

    @pytest.mark.smoke
    def regularize_test_data(self):
        point = (gs.pi + 1e-6) * gs.array(
            [[1.0, 0.0, 0.0], [2, 0.5, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
        )
        expected_2 = (
            point[1] / gs.linalg.norm(point[1]) * (gs.linalg.norm(point[1]) - 2 * gs.pi)
        )
        expected = gs.array(
            [
                [-(gs.pi - 1e-7), 0.0, 0.0],
                expected_2,
                [0.0, 0.0, 0.0],
                [(gs.pi + 1e-7) / 2.0, 0.0, 0.0],
            ]
        )

        smoke_data = [
            dict(point=point, expected=expected)
        ] + self._regularize_extreme_data()
        return smoke_data

    def _angles_close_to_pi_data(self):
        smoke_data = [
            dict(point=self.elements[angle_type])
            for angle_type in self.angles_close_to_pi
        ]

        return smoke_data

    @pytest.mark.smoke
    def group_exp_after_log_with_angles_close_to_pi_test_data(self):
        smoke_data = []
        for angle_type in self.angles_close_to_pi:
            for angle_type_base in self.elements.values():
                smoke_data.append(
                    dict(point=self.elements[angle_type], base_point=angle_type_base)
                )
        return smoke_data

    @pytest.mark.smoke
    def group_log_after_exp_with_angles_close_to_pi_test_data(self):
        smoke_data = []
        for angle_type in self.angles_close_to_pi:
            for angle_type_base in self.elements.values():
                smoke_data.append(
                    dict(
                        tangent_vec=self.elements[angle_type],
                        base_point=angle_type_base,
                    )
                )
        return smoke_data

    @pytest.mark.smoke
    def rotation_vector_and_rotation_matrix_with_angles_close_to_pi_test_data(self):
        return self._angles_close_to_pi_data()

    @pytest.mark.smoke
    def quaternion_and_rotation_vector_with_angles_close_to_pi_test_data(self):
        return self._angles_close_to_pi_data()

    @pytest.mark.smoke
    def quaternion_and_matrix_with_angles_close_to_pi_test_data(self):
        return self._angles_close_to_pi_data()

    @pytest.mark.smoke
    def compose_regularize_angles_close_to_pi_test_data(self):
        return self._angles_close_to_pi_data()

    def _tait_bryan_angles_vec_data(self, arg_name, sampler):
        data = []
        for extrinsic, zyx in itertools.product([True, False], repeat=2):
            data.extend(
                self.generate_vectorization_data(
                    arg_names=arg_name,
                    samplers={arg_name: sampler},
                    op_evaluator=lambda op, extrinsic=extrinsic, zyx=zyx, **kwargs: op(
                        **kwargs, extrinsic=extrinsic, zyx=zyx
                    ),
                    extrinsic=extrinsic,
                    zyx=zyx,
                )
            )
        return data

    def _tait_bryan_angles_random_data(self, arg_name, sampler):
        data = []
        for extrinsic, zyx in itertools.product([True, False], repeat=2):
            data.extend(
                self.generate_random_data(
                    arg_names=arg_name,
                    samplers={arg_name: sampler},
                    extrinsic=extrinsic,
                    zyx=zyx,
                )
            )
        return data

    def quaternion_from_matrix_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="rot_mat",
            samplers={"rot_mat": "random_rotation_matrix"},
        )

    def matrix_from_quaternion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="quaternion",
            samplers={"quaternion": "random_quaternion"},
        )

    def quaternion_from_rotation_vector_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="rot_vec",
            samplers={"rot_vec": "random_rotation_vector"},
        )

    def rotation_vector_from_quaternion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="quaternion",
            samplers={"quaternion": "random_quaternion"},
        )

    def matrix_from_tait_bryan_angles_vec_test_data(self):
        return self._tait_bryan_angles_vec_data(
            "tait_bryan_angles", "random_tait_bryan_angles"
        )

    def tait_bryan_angles_from_matrix_vec_test_data(self):
        return self._tait_bryan_angles_vec_data("rot_mat", "random_rotation_matrix")

    def quaternion_from_tait_bryan_angles_vec_test_data(self):
        return self._tait_bryan_angles_vec_data(
            "tait_bryan_angles", "random_tait_bryan_angles"
        )

    def tait_bryan_angles_from_quaternion_vec_test_data(self):
        return self._tait_bryan_angles_vec_data("quaternion", "random_quaternion")

    def rotation_vector_from_tait_bryan_angles_vec_test_data(self):
        return self._tait_bryan_angles_vec_data(
            "tait_bryan_angles", "random_tait_bryan_angles"
        )

    def tait_bryan_angles_from_rotation_vector_vec_test_data(self):
        return self._tait_bryan_angles_vec_data("rot_vec", "random_rotation_vector")

    def quaternion_from_matrix_after_matrix_from_quaternion_test_data(self):
        return self.generate_random_data(
            arg_names="quaternion",
            samplers={"quaternion": "random_quaternion"},
        )

    def matrix_from_quaternion_after_quaternion_from_matrix_test_data(self):
        return self.generate_random_data(
            arg_names="rot_mat",
            samplers={"rot_mat": "random_rotation_matrix"},
        )

    def quaternion_from_rotation_vector_after_rotation_vector_from_quaternion_test_data(
        self,
    ):
        return self.generate_random_data(
            arg_names="quaternion",
            samplers={"quaternion": "random_quaternion"},
        )

    def rotation_vector_from_quaternion_after_quaternion_from_rotation_vector_test_data(
        self,
    ):
        return self.generate_random_data(
            arg_names="rot_vec",
            samplers={"rot_vec": "random_rotation_vector"},
        )

    def tait_bryan_angles_from_matrix_after_matrix_from_tait_bryan_angles_test_data(
        self,
    ):
        return self._tait_bryan_angles_random_data(
            "tait_bryan_angles", "random_tait_bryan_angles"
        )

    def matrix_from_tait_bryan_angles_after_tait_bryan_angles_from_matrix_test_data(
        self,
    ):
        return self._tait_bryan_angles_random_data("rot_mat", "random_rotation_matrix")

    def quaternion_from_tait_bryan_angles_after_tait_bryan_angles_from_quaternion_test_data(
        self,
    ):
        return self._tait_bryan_angles_random_data("quaternion", "random_quaternion")

    def tait_bryan_angles_from_quaternion_after_quaternion_from_tait_bryan_angles_test_data(
        self,
    ):
        return self._tait_bryan_angles_random_data(
            "tait_bryan_angles", "random_tait_bryan_angles"
        )

    def tait_bryan_angles_from_rotation_vector_after_rotation_vector_from_tait_bryan_angles_test_data(
        self,
    ):
        return self._tait_bryan_angles_random_data(
            "tait_bryan_angles", "random_tait_bryan_angles"
        )

    def rotation_vector_from_tait_bryan_angles_after_tait_bryan_angles_from_rotation_vector_test_data(
        self,
    ):
        return self._tait_bryan_angles_random_data("rot_vec", "random_rotation_vector")
