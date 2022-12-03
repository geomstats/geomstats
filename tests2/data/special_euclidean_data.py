import pytest

import geomstats.backend as gs
from geomstats.test.data import TestData
from tests2.data.base_data import (
    LevelSetTestData,
    LieGroupTestData,
    MatrixLieAlgebraTestData,
    MatrixLieGroupTestData,
    _ProjectionMixinsTestData,
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

    def log_after_exp_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data, marks=(pytest.mark.xfail))


class SpecialEuclideanVectorsTestData(_ProjectionMixinsTestData, LieGroupTestData):
    def lie_bracket_vec_test_data(self):
        # TODO: try to understand how to make it work
        return self.generate_tests([])


class SpecialEuclideanMatrixLieAlgebraTestData(MatrixLieAlgebraTestData):
    pass


class SpecialEuclideanMatrixLieAlgebra2TestData(TestData):
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
