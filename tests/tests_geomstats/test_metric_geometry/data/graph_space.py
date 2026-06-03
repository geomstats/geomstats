from geomstats.test.data import TestData

from .quotient import AlignerAlgorithmCmpTestData


class GraphAlignerCmpTestData(AlignerAlgorithmCmpTestData):
    N_RANDOM_POINTS = [1, 2]
    trials = 5


class PointToGeodesicAlignerTestData(TestData):
    fail_for_not_implemented_errors = False

    tolerances = {"dist_along_geodesic_is_zero": {"atol": 1e-2}}

    def align_vec_test_data(self):
        return self.generate_vec_data()

    def dist_vec_test_data(self):
        return self.generate_vec_data()

    def dist_along_geodesic_is_zero_test_data(self):
        return self.generate_random_data()
