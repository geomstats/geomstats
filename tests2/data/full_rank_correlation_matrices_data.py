from tests2.data.base_data import LevelSetTestData


class FullRankCorrelationMatricesTestData(LevelSetTestData):
    def from_covariance_belongs_test_data(self):
        return self.generate_random_data()

    def from_covariance_vec_test_data(self):
        return self.generate_vec_data()

    def diag_action_vec_test_data(self):
        return self.generate_vec_data()
