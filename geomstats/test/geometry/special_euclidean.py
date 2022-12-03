import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.general_linear import SquareMatrices
from geomstats.geometry.special_euclidean import homogeneous_representation
from geomstats.test.geometry.base import LevelSetTestCase, MatrixLieAlgebraTestCase
from geomstats.test.test_case import assert_allclose
from geomstats.test.vectorization import generate_vectorization_data


def homogeneous_representation_test_case(
    rotation,
    translation,
    constant,
    expected,
    atol=gs.atol,
):
    out = homogeneous_representation(rotation, translation, constant)
    assert_allclose(out, expected, atol=atol)


def homogeneous_representation_vec_test_case(n, n_reps, atol):
    rotation = SquareMatrices(n).random_point()
    translation = Euclidean(n).random_point()
    constant = gs.array(1.0)

    expected = homogeneous_representation(rotation, translation, constant)

    vec_data = generate_vectorization_data(
        data=[
            dict(
                rotation=rotation,
                translation=translation,
                constant=constant,
                expected=expected,
                atol=atol,
            )
        ],
        arg_names=["rotation", "translation", "constant"],
        expected_name="expected",
        n_reps=n_reps,
    )
    for datum in vec_data:
        homogeneous_representation_test_case(**datum)


class SpecialEuclideanMatricesTestCase(LevelSetTestCase):
    # TODO: inherit from MatrixLieGroup
    pass


class SpecialEuclideanMatrixLieAlgebraTestCase(MatrixLieAlgebraTestCase):
    pass
