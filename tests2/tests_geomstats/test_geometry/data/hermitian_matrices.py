from .base import ComplexVectorSpaceTestData, MatrixVectorSpaceMixinsTestData


class HermitianMatricesTestData(
    MatrixVectorSpaceMixinsTestData, ComplexVectorSpaceTestData
):
    skips = (
        # due to bugs
        "to_vector_after_from_vector",
        "to_vector_and_basis",
        "from_vector_belongs",
        "from_vector_vec",
    )
