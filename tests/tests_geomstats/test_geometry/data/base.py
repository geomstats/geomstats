from .complex_manifold import ComplexManifoldTestData
from .manifold import ManifoldTestData
from .mixins import ProjectionMixinsTestData


class _VectorSpaceMixinsTestData(ProjectionMixinsTestData):
    def random_point_is_tangent_test_data(self):
        return self.generate_random_data(arg_names="point")

    def to_tangent_is_projection_test_data(self):
        return self.generate_random_data(arg_names="point")


class VectorSpaceTestData(_VectorSpaceMixinsTestData, ManifoldTestData):
    def basis_cardinality_test_data(self):
        return [dict()]

    def basis_belongs_test_data(self):
        return [dict()]


class ComplexVectorSpaceTestData(_VectorSpaceMixinsTestData, ComplexManifoldTestData):
    pass


class MatrixVectorSpaceTestData(VectorSpaceTestData):
    def basis_representation_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="matrix_representation",
            samplers={"matrix_representation": "random_point"},
        )

    def matrix_representation_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="basis_representation",
            samplers={"basis_representation": "random_basis_representation"},
        )

    def basis_representation_and_basis_test_data(self):
        return self.generate_random_data(
            arg_names="matrix_representation",
            samplers={"matrix_representation": "random_point"},
        )

    def matrix_representation_belongs_test_data(self):
        return self.generate_random_data(
            arg_names="basis_representation",
            samplers={"basis_representation": "random_basis_representation"},
        )

    def matrix_representation_after_basis_representation_test_data(self):
        return self.generate_random_data(
            arg_names="matrix_representation",
            samplers={"matrix_representation": "random_point"},
        )

    def basis_representation_after_matrix_representation_test_data(self):
        return self.generate_random_data(
            arg_names="basis_representation",
            samplers={"basis_representation": "random_basis_representation"},
        )


class ComplexMatrixVectorSpaceTestData(ComplexVectorSpaceTestData):
    pass


class LevelSetTestData(ProjectionMixinsTestData, ManifoldTestData):
    def submersion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="point",
        )

    def tangent_submersion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("point", "vector"),
        )

    def submersion_is_zero_test_data(self):
        return self.generate_random_data(arg_names="point")

    def tangent_submersion_is_zero_test_data(self):
        return self.generate_random_data(arg_names=("point", "vector"))


class _OpenSetMixinsTestData(ProjectionMixinsTestData):
    def to_tangent_is_tangent_in_embedding_space_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))


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
        return self.generate_vectorization_data(arg_names="point")

    def tangent_immersion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def jacobian_immersion_vec_test_data(self):
        return self.generate_vectorization_data(arg_names="base_point")

    def hessian_immersion_vec_test_data(self):
        return self.generate_vectorization_data(arg_names="base_point")

    def immersion_belongs_test_data(self):
        return self.generate_random_data(arg_names="point")

    def tangent_immersion_is_tangent_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))


class DiffeomorphicManifoldTestData(ManifoldTestData):
    pass
