from geomstats.test.data import TestData

from .complex_manifold import ComplexManifoldTestData
from .manifold import ManifoldTestData


class _ProjectionMixinsTestData:
    def projection_vec_test_data(self):
        return self.generate_vec_data()

    def projection_belongs_test_data(self):
        return self.generate_random_data()


class _VectorSpaceMixinsTestData(_ProjectionMixinsTestData):
    def basis_cardinality_test_data(self):
        return None

    def basis_belongs_test_data(self):
        return self.generate_tests([dict()])

    def random_point_is_tangent_test_data(self):
        return self.generate_random_data()

    def to_tangent_is_projection_test_data(self):
        return self.generate_random_data()


class VectorSpaceTestData(_VectorSpaceMixinsTestData, ManifoldTestData):
    pass


class ComplexVectorSpaceTestData(_VectorSpaceMixinsTestData, ComplexManifoldTestData):
    pass


class MatrixVectorSpaceMixinsTestData(TestData):
    def to_vector_vec_test_data(self):
        return self.generate_vec_data()

    def to_vector_and_basis_test_data(self):
        return self.generate_random_data()

    def from_vector_vec_test_data(self):
        return self.generate_vec_data()

    def from_vector_belongs_test_data(self):
        return self.generate_random_data()

    def from_vector_after_to_vector_test_data(self):
        return self.generate_random_data()

    def to_vector_after_from_vector_test_data(self):
        return self.generate_random_data()


ComplexMatrixVectorSpaceMixinsTestData = MatrixVectorSpaceMixinsTestData


class LevelSetTestData(_ProjectionMixinsTestData, ManifoldTestData):
    def submersion_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_submersion_vec_test_data(self):
        return self.generate_vec_data()


class _OpenSetMixinsTestData(_ProjectionMixinsTestData):
    def to_tangent_is_tangent_in_embedding_space_test_data(self):
        return self.generate_random_data()


class OpenSetTestData(_OpenSetMixinsTestData, ManifoldTestData):
    pass


class ComplexOpenSetTestData(_OpenSetMixinsTestData, ComplexManifoldTestData):
    pass
