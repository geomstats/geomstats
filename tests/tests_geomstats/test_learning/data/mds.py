from ._base import BaseEstimatorTestData, TestData


class PairwiseDistsTestData(TestData):
    def dists_among_selves_test_data(self):
        return self.generate_random_data()

    def one_point_test_data(self):
        return self.generate_random_data()

    def symmetric_test_data(self):
        return self.generate_random_data()

    def general_test_data(self):
        return self.generate_random_data()


class MDSTestData(BaseEstimatorTestData):
    def minimal_fit_test_data(self):
        return self.generate_random_data()

    def minimal_fit_transform_test_data(self):
        return self.generate_random_data()
