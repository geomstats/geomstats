from polpo.testing.geometric import GeometricMapCaseData


class UniqueDiagonalMatrixAlgorithmTestData(GeometricMapCaseData):
    def map_belongs_to_image_test_data(self):
        return self.generate_random_data(
            arg_names="point",
        )

    def map_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="point",
            op_name="__call__",
        )


class SPDScalingFinderTestData(GeometricMapCaseData):
    def rows_sum_to_one_test_data(self):
        return self.generate_random_data(
            arg_names="point",
        )

    def values_are_positive_test_data(self):
        return self.generate_random_data(
            arg_names="point",
        )

    def map_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="point",
            op_name="__call__",
        )
