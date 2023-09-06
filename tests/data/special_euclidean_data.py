import random
from contextlib import nullcontext as does_not_raise

import pytest

import geomstats.backend as gs
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.special_euclidean import (
    SpecialEuclidean,
    SpecialEuclideanMatricesCanonicalLeftMetric,
    SpecialEuclideanMatricesLieAlgebra,
)
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.data_generation import (
    TestData,
    _InvariantMetricTestData,
    _LieGroupTestData,
    _MatrixLieAlgebraTestData,
)


def group_useful_matrix(theta, elem_33=1.0):
    return gs.array(
        [
            [gs.cos(theta), -gs.sin(theta), 2.0],
            [gs.sin(theta), gs.cos(theta), 3.0],
            [0.0, 0.0, elem_33],
        ]
    )


def algebra_useful_matrix(theta, elem_33=0.0):
    return gs.array([[0.0, -theta, 2.0], [theta, 0.0, 3.0], [0.0, 0.0, elem_33]])


point_1 = gs.array([0.1, 0.2, 0.3])
point_2 = gs.array([0.5, 5.0, 60.0])

translation_large = gs.array([0.0, 5.0, 6.0])
translation_small = gs.array([0.0, 0.6, 0.7])

elements_all = {
    "translation_large": translation_large,
    "translation_small": translation_small,
    "point_1": point_1,
    "point_2": point_2,
}
elements = elements_all

elements_matrices_all = {
    key: SpecialEuclidean(2, point_type="vector").matrix_from_vector(elements_all[key])
    for key in elements_all
}
elements_matrices = elements_matrices_all


class SpecialEuclideanTestData(_LieGroupTestData):
    Space = SpecialEuclidean

    n_list = random.sample(range(2, 4), 2)
    space_args_list = [(n,) for n in n_list] + [(2, "vector"), (3, "vector")]
    shape_list = [(n + 1, n + 1) for n in n_list] + [(3,)] + [(6,)]
    n_tangent_vecs_list = [2, 3] * 2
    n_points_list = [2, 3] * 2
    n_vecs_list = [2, 3] * 2

    def belongs_test_data(self):
        smoke_data = [
            dict(n=2, mat=group_useful_matrix(gs.pi / 3, elem_33=1.0), expected=True),
            dict(n=2, mat=group_useful_matrix(gs.pi / 3, elem_33=0.0), expected=False),
            dict(
                n=2,
                mat=gs.stack(
                    [
                        group_useful_matrix(gs.pi / 3, elem_33=1.0),
                        group_useful_matrix(gs.pi / 3, elem_33=0.0),
                    ]
                ),
                expected=[True, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def identity_test_data(self):
        smoke_data = [
            dict(n=2, expected=gs.eye(3)),
            dict(n=3, expected=gs.eye(4)),
            dict(n=10, expected=gs.eye(11)),
        ]
        return self.generate_tests(smoke_data)

    def is_tangent_test_data(self):
        theta = gs.pi / 3
        vec_1 = gs.array([[0.0, -theta, 2.0], [theta, 0.0, 3.0], [0.0, 0.0, 0.0]])
        vec_2 = gs.array([[0.0, -theta, 2.0], [theta, 0.0, 3.0], [0.0, 0.0, 1.0]])
        point = group_useful_matrix(theta)
        smoke_data = [
            dict(n=2, tangent_vec=point @ vec_1, base_point=point, expected=True),
            dict(n=2, tangent_vec=point @ vec_2, base_point=point, expected=False),
            dict(
                n=2,
                tangent_vec=gs.stack([point @ vec_1, point @ vec_2]),
                base_point=point,
                expected=[True, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def basis_representation_test_data(self):
        n_list = random.sample(range(2, 50), 10)
        n_samples = 100
        random_data = [
            dict(n=n, vec=gs.random.rand(n_samples, self.group.dim)) for n in n_list
        ]
        return self.generate_tests([], random_data)

    def inverse_shape_test_data(self):
        n_list = random.sample(range(2, 50), 10)
        n_samples = 10
        random_data = [
            dict(
                n=n,
                points=SpecialEuclidean(n).random_point(n_samples),
                expected=(n_samples, n + 1, n + 1),
            )
            for n in n_list
        ]
        return self.generate_tests([], random_data)

    def compose_shape_test_data(self):
        n_list = random.sample(range(2, 50), 10)
        n_samples = 10
        random_data = [
            dict(
                n=n,
                point_a=SpecialEuclidean(n).random_point(n_samples),
                point_b=SpecialEuclidean(n).random_point(n_samples),
                expected=(n_samples, n + 1, n + 1),
            )
            for n in n_list
        ]
        random_data += [
            dict(
                n=n,
                point_a=SpecialEuclidean(n).random_point(),
                point_b=SpecialEuclidean(n).random_point(n_samples),
                expected=(n_samples, n + 1, n + 1),
            )
            for n in n_list
        ]
        random_data += [
            dict(
                n=n,
                point_a=SpecialEuclidean(n).random_point(n_samples),
                point_b=SpecialEuclidean(n).random_point(),
                expected=(n_samples, n + 1, n + 1),
            )
            for n in n_list
        ]
        return self.generate_tests([], random_data)

    def regularize_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point_type="vector",
                point=elements_all["point_1"],
                expected=elements_all["point_1"],
            )
        ]
        return self.generate_tests(smoke_data)

    def regularize_shape_test_data(self):
        smoke_data = [dict(n=2, point_type="vector", n_samples=3)]
        return self.generate_tests(smoke_data)

    def compose_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point_type="vector",
                point_1=elements_all["translation_small"],
                point_2=elements_all["translation_large"],
                expected=elements_all["translation_small"]
                + elements_all["translation_large"],
            )
        ]
        return self.generate_tests(smoke_data)

    def group_exp_from_identity_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point_type="vector",
                tangent_vec=elements_all["translation_small"],
                expected=elements_all["translation_small"],
            ),
            dict(
                n=2,
                point_type="vector",
                tangent_vec=gs.stack([elements_all["translation_small"]] * 2),
                expected=gs.stack([elements_all["translation_small"]] * 2),
            ),
        ]
        return self.generate_tests(smoke_data)

    def group_log_from_identity_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point_type="vector",
                point=elements_all["translation_small"],
                expected=elements_all["translation_small"],
            ),
            dict(
                n=2,
                point_type="vector",
                point=gs.stack([elements_all["translation_small"]] * 2),
                expected=gs.stack([elements_all["translation_small"]] * 2),
            ),
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=100.0)


class SpecialEuclideanMatricesLieAlgebraTestData(_MatrixLieAlgebraTestData):
    Space = SpecialEuclideanMatricesLieAlgebra

    n_list = random.sample(range(2, 5), 2)
    space_args_list = [(n,) for n in n_list]
    shape_list = [(n + 1, n + 1) for n in n_list]
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    def belongs_test_data(self):
        theta = gs.pi / 3
        smoke_data = [
            dict(n=2, vec=algebra_useful_matrix(theta, elem_33=0.0), expected=True),
            dict(n=2, vec=algebra_useful_matrix(theta, elem_33=1.0), expected=False),
            dict(
                n=2,
                vec=gs.stack(
                    [
                        algebra_useful_matrix(theta, elem_33=0.0),
                        algebra_useful_matrix(theta, elem_33=1.0),
                    ]
                ),
                expected=[True, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def dim_test_data(self):
        smoke_data = [
            dict(n=2, expected=3),
            dict(n=3, expected=6),
            dict(n=10, expected=55),
        ]
        return self.generate_tests(smoke_data)


class SpecialEuclideanMatricesCanonicalLeftMetricTestData(_InvariantMetricTestData):
    n_list = random.sample(range(2, 5), 2)

    metric_args_list = [{} for _ in n_list]
    shape_list = [(n + 1, n + 1) for n in n_list]
    group_list = space_list = [SpecialEuclidean(n, equip=False) for n in n_list]

    n_points_list = [2, 3]
    n_tangent_vecs_list = [2, 3]
    n_points_a_list = [2, 3]
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = SpecialEuclideanMatricesCanonicalLeftMetric

    def left_metric_wrong_group_test_data(self):
        smoke_data = [
            dict(group=SpecialEuclidean(2), expected=does_not_raise()),
            dict(group=SpecialEuclidean(3), expected=does_not_raise()),
            dict(
                group=SpecialEuclidean(2, point_type="vector"),
                expected=pytest.raises(ValueError),
            ),
            dict(group=SpecialOrthogonal(3), expected=pytest.raises(ValueError)),
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=10.0)

    def log_after_exp_at_identity_test_data(self):
        return super().log_after_exp_at_identity_test_data(amplitude=10.0)


class SpecialEuclideanMatrixCanonicalRightMetricTestData(_InvariantMetricTestData):
    n_list = [2]

    metric_args_list = [{"left": False} for _ in n_list]
    shape_list = [(n + 1, n + 1) for n in n_list]
    group_list = space_list = [SpecialEuclidean(n, equip=False) for n in n_list]

    n_points_list = random.sample(range(1, 3), 1)
    n_tangent_vecs_list = random.sample(range(1, 3), 1)
    n_points_a_list = random.sample(range(1, 3), 1)
    n_points_b_list = [1]
    alpha_list = [1] * 1
    n_rungs_list = [1] * 1
    scheme_list = ["pole"] * 1

    Metric = InvariantMetric

    tolerances = {
        "exp_after_log_at_identity": {"atol": 1e-2},
        "log_after_exp_at_identity": {"atol": 1e-1},
    }

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=100.0)

    def log_after_exp_at_identity_test_data(self):
        return super().log_after_exp_at_identity_test_data(amplitude=100.0)

    def right_exp_coincides_test_data(self):
        smoke_data = [
            dict(
                n=2,
                initial_vec=gs.array([gs.pi / 2, 1.0, 1.0]),
            )
        ]
        return self.generate_tests(smoke_data)


class SpecialEuclidean3VectorsTestData(TestData):
    group = SpecialEuclidean(n=3, point_type="vector", equip=False)

    angle_0 = gs.zeros(6)
    angle_close_0 = 1e-10 * gs.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0]) + gs.array(
        [0.0, 0.0, 0.0, 1.0, 5.0, 2]
    )
    angle_close_pi_low = (gs.pi - 1e-9) / gs.sqrt(2.0) * gs.array(
        [0.0, 1.0, -1.0, 0.0, 0.0, 0.0]
    ) + gs.array([0.0, 0.0, 0.0, -100.0, 0.0, 2.0])
    angle_pi = gs.pi / gs.sqrt(3.0) * gs.array(
        [1.0, 1.0, -1.0, 0.0, 0.0, 0.0]
    ) + gs.array([0.0, 0.0, 0.0, -10.2, 0.0, 2.6])
    angle_close_pi_high = (gs.pi + 1e-9) / gs.sqrt(3.0) * gs.array(
        [-1.0, 1.0, -1.0, 0.0, 0.0, 0.0]
    ) + gs.array([0.0, 0.0, 0.0, -100.0, 0.0, 2.0])
    angle_in_pi_2pi = (gs.pi + 0.3) / gs.sqrt(5.0) * gs.array(
        [-2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ) + gs.array([0.0, 0.0, 0.0, -100.0, 0.0, 2.0])
    angle_close_2pi_low = (2 * gs.pi - 1e-9) / gs.sqrt(6.0) * gs.array(
        [2.0, 1.0, -1.0, 0.0, 0.0, 0.0]
    ) + gs.array([0.0, 0.0, 0.0, 8.0, 555.0, -2.0])
    angle_2pi = 2.0 * gs.pi / gs.sqrt(3.0) * gs.array(
        [1.0, 1.0, -1.0, 0.0, 0.0, 0.0]
    ) + gs.array([0.0, 0.0, 0.0, 1.0, 8.0, -10.0])
    angle_close_2pi_high = (2.0 * gs.pi + 1e-9) / gs.sqrt(2.0) * gs.array(
        [1.0, 0.0, -1.0, 0.0, 0.0, 0.0]
    ) + gs.array([0.0, 0.0, 0.0, 1.0, 8.0, -10.0])

    point_1 = gs.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    point_2 = gs.array([0.5, 0.0, -0.3, 0.4, 5.0, 60.0])

    translation_large = gs.array([0.0, 0.0, 0.0, 0.4, 0.5, 0.6])
    translation_small = gs.array([0.0, 0.0, 0.0, 0.5, 0.6, 0.7])
    rot_with_parallel_trans = gs.array([gs.pi / 3.0, 0.0, 0.0, 1.0, 0.0, 0.0])

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
        "translation_large": translation_large,
        "translation_small": translation_small,
        "point_1": point_1,
        "point_2": point_2,
        "rot_with_parallel_trans": rot_with_parallel_trans,
    }

    # Metrics - only diagonals
    diag_mat_at_identity = gs.eye(6) * gs.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])

    # FIXME:
    # 'left': left_metric,
    # 'right': right_metric}
    metric_args_list = [
        dict(metric_mat_at_identity=gs.eye(group.dim), left=True),
        dict(metric_mat_at_identity=gs.eye(group.dim), left=False),
        dict(metric_mat_at_identity=diag_mat_at_identity, left=True),
        dict(metric_mat_at_identity=diag_mat_at_identity, left=True),
    ]

    angles_close_to_pi = [
        "angle_close_pi_low",
        "angle_pi",
        "angle_close_pi_high",
    ]

    tolerances = {
        "log_after_exp": {"atol": 1e-6},
        "exp_after_log": {"atol": 1e-4},
        "log_after_exp_with_angles_close_to_pi": {"atol": 1e-6},
        "exp_after_log_with_angles_close_to_pi": {"atol": 1e-8},
        "exp_after_log_right_with_angles_close_to_pi": {"atol": 1e-8},
    }

    Metric = InvariantMetric

    def exp_after_log_right_with_angles_close_to_pi_test_data(self):
        smoke_data = []
        for metric_args in self.metric_args_list + [{}]:
            for base_point in self.elements.values():
                for element_type in self.angles_close_to_pi:
                    point = self.elements[element_type]
                    smoke_data.append(
                        dict(
                            metric_args=metric_args,
                            point=point,
                            base_point=base_point,
                        )
                    )
        return self.generate_tests(smoke_data)

    def exp_after_log_test_data(self):
        smoke_data = []
        for metric_args in self.metric_args_list + [{}]:
            for base_point in self.elements.values():
                for element_type in self.elements:
                    if element_type in self.angles_close_to_pi:
                        continue
                    point = self.elements[element_type]
                    smoke_data.append(
                        dict(
                            metric_args=metric_args,
                            point=point,
                            base_point=base_point,
                        )
                    )
        return self.generate_tests(smoke_data)

    def log_after_exp_with_angles_close_to_pi_test_data(self):
        smoke_data = []
        for metric_args in self.metric_args_list:
            for base_point in self.elements.values():
                for element_type in self.angles_close_to_pi:
                    tangent_vec = self.elements[element_type]
                    smoke_data.append(
                        dict(
                            metric_args=metric_args,
                            tangent_vec=tangent_vec,
                            base_point=base_point,
                        )
                    )
        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        smoke_data = []
        for metric_args in self.metric_args_list:
            if metric_args["left"] is False:
                continue

            for base_point in self.elements.values():
                for element_type in self.elements:
                    if element_type in self.angles_close_to_pi:
                        continue
                    tangent_vec = self.elements[element_type]
                    smoke_data.append(
                        dict(
                            metric_args=metric_args,
                            tangent_vec=tangent_vec,
                            base_point=base_point,
                        )
                    )
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        rot_vec_base_point = gs.array([0.0, 0.0, 0.0])
        translation_base_point = gs.array([4.0, -1.0, 10000.0])
        transfo_base_point = gs.concatenate(
            [rot_vec_base_point, translation_base_point], axis=0
        )

        # Tangent vector is a translation (no infinitesimal rotational part)
        # Expect the sum of the translation
        # with the translation of the reference point
        rot_vec = gs.array([0.0, 0.0, 0.0])
        translation = gs.array([1.0, 0.0, -3.0])
        tangent_vec = gs.concatenate([rot_vec, translation], axis=0)
        expected = gs.concatenate(
            [gs.array([0.0, 0.0, 0.0]), gs.array([5.0, -1.0, 9997.0])], axis=0
        )
        smoke_data = [
            dict(
                metric_args=self.metric_args_list[0],  # left_canonical
                tangent_vec=tangent_vec,
                base_point=transfo_base_point,
                expected=expected,
            ),
            dict(
                metric_args={},
                tangent_vec=self.elements["translation_small"],
                base_point=self.elements["translation_large"],
                expected=self.elements["translation_large"]
                + self.elements["translation_small"],
            ),
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        rot_vec_base_point = gs.array([0.0, 0.0, 0.0])
        translation_base_point = gs.array([4.0, 0.0, 0.0])
        transfo_base_point = gs.concatenate(
            [rot_vec_base_point, translation_base_point], axis=0
        )

        # Point is a translation (no rotational part)
        # Expect the difference of the translation
        # by the translation of the reference point
        rot_vec = gs.array([0.0, 0.0, 0.0])
        translation = gs.array([-1.0, -1.0, -1.2])
        point = gs.concatenate([rot_vec, translation], axis=0)

        expected = gs.concatenate(
            [gs.array([0.0, 0.0, 0.0]), gs.array([-5.0, -1.0, -1.2])], axis=0
        )
        smoke_data = [
            dict(
                metric_args=self.metric_args_list[0],  # left_canonical
                point=point,
                base_point=transfo_base_point,
                expected=expected,
            ),
            dict(
                metric_args={},
                point=self.elements["translation_large"],
                base_point=self.elements["translation_small"],
                expected=self.elements["translation_large"]
                - self.elements["translation_small"],
            ),
        ]
        return self.generate_tests(smoke_data)

    def regularize_extreme_cases_test_data(self):
        smoke_data = []
        for angle_type in ["angle_close_0", "angle_close_pi_low", "angle_0"]:
            point = self.elements[angle_type]
            smoke_data += [dict(point=point, expected=point)]

        angle_type = "angle_pi"
        point = self.elements[angle_type]
        smoke_data += [dict(point=point, expected=point)]

        angle_type = "angle_close_pi_high"
        point = self.elements[angle_type]

        norm = gs.linalg.norm(point[:3])
        expected_rot = gs.concatenate(
            [point[:3] / norm * (norm - 2 * gs.pi), gs.zeros(3)], axis=0
        )
        expected_trans = gs.concatenate([gs.zeros(3), point[3:6]], axis=0)
        expected = expected_rot + expected_trans
        smoke_data += [dict(point=point, expected=expected)]

        in_pi_2pi = ["angle_in_pi_2pi", "angle_close_2pi_low"]

        for angle_type in in_pi_2pi:
            point = self.elements[angle_type]
            angle = gs.linalg.norm(point[:3])
            new_angle = gs.pi - (angle - gs.pi)

            expected_rot = gs.concatenate(
                [-new_angle * (point[:3] / angle), gs.zeros(3)], axis=0
            )
            expected_trans = gs.concatenate([gs.zeros(3), point[3:6]], axis=0)
            expected = expected_rot + expected_trans
            smoke_data += [dict(point=point, expected=expected)]

        angle_type = "angle_2pi"
        point = self.elements[angle_type]

        expected = gs.concatenate([gs.zeros(3), point[3:6]], axis=0)
        smoke_data += [dict(point=point, expected=expected)]

        angle_type = "angle_close_2pi_high"
        point = self.elements[angle_type]
        angle = gs.linalg.norm(point[:3])
        new_angle = angle - 2 * gs.pi

        expected_rot = gs.concatenate(
            [new_angle * point[:3] / angle, gs.zeros(3)], axis=0
        )
        expected_trans = gs.concatenate([gs.zeros(3), point[3:6]], axis=0)
        expected = expected_rot + expected_trans
        smoke_data += [dict(point=point, expected=expected)]
        return self.generate_tests(smoke_data)
