from tests2.data.base_data import OpenSetTestData


class PositiveLowerTriangularMatricesTestData(OpenSetTestData):
    def gram_vec_test_data(self):
        return self.generate_vec_data()

    def differential_gram_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_differential_gram_vec_test_data(self):
        return self.generate_vec_data()

    def gram_belongs_to_spd_matrices_test_data(self):
        return self.generate_random_data()

    def differential_gram_belongs_to_symmetric_matrices_test_data(self):
        return self.generate_random_data()

    def inverse_differential_gram_belongs_to_lower_triangular_matrices_test_data(self):
        return self.generate_random_data()
