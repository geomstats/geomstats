import random

from geomstats.test.data import TestData


class RiemannianKMeansOneClusterTestData(TestData):
    def one_cluster_test_data(self):
        return self.generate_tests([dict(n_points=random.randint(2, 10))])


class RiemannianKMeansTestData(TestData):
    def cluster_assignment_test_data(self):
        return self.generate_tests([dict(n_points=random.randint(5, 10))])

    def centroids_belong_test_data(self):
        return self.generate_tests([dict(n_points=random.randint(5, 10))])

    def centroids_shape_test_data(self):
        return self.generate_tests([dict(n_points=random.randint(5, 10))])
