import random

from geomstats.test.data import TestData


class BaseEstimatorTestData(TestData):
    MIN_RANDOM = 2
    MAX_RANDOM = 5

    def generate_random_data(self, marks=()):
        return self.generate_tests(
            [dict(n_points=random.randint(self.MIN_RANDOM, self.MAX_RANDOM))],
            marks=marks,
        )


class MeanEstimatorMixinsTestData:
    def one_point_test_data(self):
        return self.generate_tests([{}])

    def n_times_same_point_test_data(self):
        return self.generate_tests(
            [dict(n_reps=random.randint(self.MIN_RANDOM, self.MAX_RANDOM))]
        )

    def estimate_belongs_test_data(self):
        return self.generate_random_data()


class ClusterMixinsTestData:
    def n_repeated_clusters_test_data(self):
        return self.generate_tests(
            [dict(n_reps=random.randint(self.MIN_RANDOM, self.MAX_RANDOM))]
        )

    def cluster_assignment_test_data(self):
        return self.generate_random_data()

    def cluster_centers_belong_test_data(self):
        return self.generate_random_data()

    def cluster_centers_shape_test_data(self):
        return self.generate_random_data()
