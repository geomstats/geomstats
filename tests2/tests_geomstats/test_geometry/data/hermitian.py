from tests2.data.base_data import ComplexVectorSpaceTestData


class HermitianTestData(ComplexVectorSpaceTestData):
    def exp_vec_test_data(self):
        return self.generate_vec_data()

    def exp_random_test_data(self):
        return self.generate_random_data()

    def identity_belongs_test_data(self):
        return self.generate_tests([dict()])
