import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import LevelSetTestData
from .invariant_metric import InvariantMetricMatrixTestData
from .lie_group import LieGroupTestData, MatrixLieGroupTestData
from .mixins import ProjectionMixinsTestData

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


def homogeneous_representation_test_data():
    rotation = gs.ones((2, 2)) * 2
    translation = gs.ones((2,)) * 3
    constant = 1.0
    expected = gs.array([[2.0, 2.0, 3.0], [2.0, 2.0, 3.0], [0.0, 0.0, 1.0]])

    data = [[rotation, translation, constant, expected]]

    return data


class SpecialEuclideanMatricesTestData(MatrixLieGroupTestData, LevelSetTestData):
    tolerances = {
        "projection_belongs": {"atol": 1e-4},
    }


class SpecialEuclideanMatrices2TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=group_useful_matrix(gs.pi / 3, elem_33=1.0), expected=True),
            dict(point=group_useful_matrix(gs.pi / 3, elem_33=0.0), expected=False),
            dict(
                point=gs.stack(
                    [
                        group_useful_matrix(gs.pi / 3, elem_33=1.0),
                        group_useful_matrix(gs.pi / 3, elem_33=0.0),
                    ]
                ),
                expected=[True, False],
            ),
        ]
        return self.generate_tests(data)

    def identity_test_data(self):
        data = [
            dict(expected=gs.eye(3)),
        ]

        return self.generate_tests(data)

    def is_tangent_test_data(self):
        theta = gs.pi / 3
        vec_1 = gs.array([[0.0, -theta, 2.0], [theta, 0.0, 3.0], [0.0, 0.0, 0.0]])
        vec_2 = gs.array([[0.0, -theta, 2.0], [theta, 0.0, 3.0], [0.0, 0.0, 1.0]])
        point = group_useful_matrix(theta)
        data = [
            dict(vector=point @ vec_1, base_point=point, expected=True),
            dict(vector=point @ vec_2, base_point=point, expected=False),
            dict(
                vector=gs.stack([point @ vec_1, point @ vec_2]),
                base_point=point,
                expected=[True, False],
            ),
        ]
        return self.generate_tests(data)


class SpecialEuclideanVectorsTestData(ProjectionMixinsTestData, LieGroupTestData):
    fail_for_not_implemented_errors = False

    def matrix_from_vector_vec_test_data(self):
        return self.generate_vec_data()

    def matrix_from_vector_belongs_to_matrices_test_data(self):
        return self.generate_random_data()

    def regularize_tangent_vec_vec_test_data(self):
        return self.generate_vec_data()

    def regularize_tangent_vec_at_identity_vec_test_data(self):
        return self.generate_vec_data()


class SpecialEuclidean2VectorsTestData(TestData):
    def regularize_test_data(self):
        data = [
            dict(
                point=elements_all["point_1"],
                expected=elements_all["point_1"],
            )
        ]
        return self.generate_tests(data)

    def compose_test_data(self):
        data = [
            dict(
                point_a=elements_all["translation_small"],
                point_b=elements_all["translation_large"],
                expected=elements_all["translation_small"]
                + elements_all["translation_large"],
            )
        ]
        return self.generate_tests(data)

    def exp_from_identity_test_data(self):
        data = [
            dict(
                tangent_vec=elements_all["translation_small"],
                expected=elements_all["translation_small"],
            ),
            dict(
                tangent_vec=gs.stack([elements_all["translation_small"]] * 2),
                expected=gs.stack([elements_all["translation_small"]] * 2),
            ),
        ]
        return self.generate_tests(data)

    def log_from_identity_test_data(self):
        data = [
            dict(
                point=elements_all["translation_small"],
                expected=elements_all["translation_small"],
            ),
            dict(
                point=gs.stack([elements_all["translation_small"]] * 2),
                expected=gs.stack([elements_all["translation_small"]] * 2),
            ),
        ]
        return self.generate_tests(data)


class SpecialEuclideanMatricesLieAlgebra2TestData(TestData):
    def belongs_test_data(self):
        theta = gs.pi / 3
        data = [
            dict(
                point=algebra_useful_matrix(theta, elem_33=0.0), expected=gs.array(True)
            ),
            dict(
                point=algebra_useful_matrix(theta, elem_33=1.0),
                expected=gs.array(False),
            ),
            dict(
                point=gs.stack(
                    [
                        algebra_useful_matrix(theta, elem_33=0.0),
                        algebra_useful_matrix(theta, elem_33=1.0),
                    ]
                ),
                expected=gs.array([True, False]),
            ),
        ]
        return self.generate_tests(data)


class SpecialEuclideanMatricesCanonicalLeftMetricTestData(
    InvariantMetricMatrixTestData
):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    skips = ("exp_at_identity_vec",)
