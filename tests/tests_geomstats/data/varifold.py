from geomstats.test.data import TestData

from ..test_geometry.data.mixins import DistMixinsTestData


class KernelTestData(TestData):
    N_RANDOM_POINTS = [1]
    trials = 1

    def against_other_random_test_data(self):
        return self.generate_random_data()


class VarifoldMetricTestData(DistMixinsTestData, TestData):
    skip_vec = True
    trials = 1
    N_RANDOM_POINTS = [1]
