import random

from geomstats.test.data import TestData


class ForwardDifferenceTestData(TestData):
    N_TIME_POINTS = random.sample(range(5, 10), 1)
    trials = 1

    def forward_difference_last_test_data(self):
        return self.generate_random_data_with_time()


class CenteredDifferenceTestData(TestData):
    N_TIME_POINTS = random.sample(range(5, 10), 1)
    trials = 1

    def centered_difference_random_index_test_data(self, marks=()):
        data = []
        for n_points in self.N_RANDOM_POINTS:
            for n_times in self.N_TIME_POINTS:
                for endpoints in [True, False]:
                    data.append(
                        dict(n_points=n_points, n_times=n_times, endpoints=endpoints)
                    )

        return self.generate_tests(data, marks=marks)

        return self.generate_random_data_with_time()


class SecondCenteredDifferenceTestData(TestData):
    N_TIME_POINTS = random.sample(range(5, 10), 1)
    trials = 1

    def second_centered_difference_random_index_test_data(self):
        return self.generate_random_data_with_time()
