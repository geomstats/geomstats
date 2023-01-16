from tests2.data.base_data import VectorSpaceTestData


class EuclideanTestData(VectorSpaceTestData):
    def exp_vec_test_data(self):
        return self.generate_vec_data()

    def exp_random_test_data(self):
        return self.generate_random_data()
