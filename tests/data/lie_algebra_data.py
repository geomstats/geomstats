import geomstats.backend as gs
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.data_generation import TestData


class TestDataLieAlgebra(TestData):
    def dimension_test_data(self):
        smoke_data = [dict(algebra=SkewSymmetricMatrices(4), expected=6)]
        return self.generate_tests(smoke_data)

    def matrix_representation_and_belongs_test_data(self):
        smoke_data = [
            dict(algebra=SkewSymmetricMatrices(4), point=gs.random.rand(2, 6))
        ]
        return self.generate_tests(smoke_data)

    def orthonormal_basis_test_data(self):
        smoke_data = [
            dict(group=SpecialOrthogonal(3), metric_mat_at_identity=None),
            dict(
                group=SpecialOrthogonal(3),
                metric_mat_at_identity=from_vector_to_diagonal_matrix(
                    gs.array([1.0, 2.0, 3.0])
                ),
            ),
        ]
        return self.generate_tests(smoke_data)

    def orthonormal_basis_se3_test_data(self):
        smoke_data = [
            dict(group=SpecialEuclidean(3), metric_mat_at_identity=None),
            dict(
                group=SpecialEuclidean(3),
                metric_mat_at_identity=from_vector_to_diagonal_matrix(
                    gs.cast(gs.arange(1, SpecialEuclidean(3).dim + 1), gs.float32)
                ),
            ),
        ]
        return self.generate_tests(smoke_data)
