from geomstats.test.data import TestData

from .complex_manifold import ComplexManifoldTestData
from .manifold import ManifoldTestData
from .mixins import ProjectionMixinsTestData


class _VectorSpaceMixinsTestData(ProjectionMixinsTestData):
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


class LevelSetTestData(ProjectionMixinsTestData, ManifoldTestData):
    def submersion_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_submersion_vec_test_data(self):
        return self.generate_vec_data()


class _OpenSetMixinsTestData(ProjectionMixinsTestData):
    def to_tangent_is_tangent_in_embedding_space_test_data(self):
        return self.generate_random_data()


class OpenSetTestData(_OpenSetMixinsTestData, ManifoldTestData):
    pass


class VectorSpaceOpenSetTestData(OpenSetTestData):
    pass


class ComplexVectorSpaceOpenSetTestData(
    _OpenSetMixinsTestData, ComplexManifoldTestData
):
    pass


class ImmersedSetTestData(ProjectionMixinsTestData, ManifoldTestData):
    def immersion_vec_test_data(self):
        return self.generate_vec_data()

    def immersion_belongs_test_data(self):
        return self.generate_random_data()

    def tangent_immersion_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_immersion_is_tangent_test_data(self):
        return self.generate_random_data()

    def jacobian_immersion_vec_test_data(self):
        return self.generate_vec_data()

    def hessian_immersion_vec_test_data(self):
        return self.generate_vec_data()


class DiffeomorphicManifoldTestData(ManifoldTestData):
    pass
