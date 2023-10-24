import random

from geomstats.test.data import TestData


class InformationManifoldMixinTestData(TestData):
    N_SAMPLE_POINTS = [1] + random.sample(range(2, 5), 1)

    def generate_random_data_with_samples(self, marks=()):
        data = []
        for n_points in self.N_RANDOM_POINTS:
            for n_samples in self.N_SAMPLE_POINTS:
                data.append(dict(n_points=n_points, n_samples=n_samples))

        return self.generate_tests(data, marks=marks)

    def generate_repeated_data_with_samples(self, marks=()):
        data = []
        for n_reps in self.N_VEC_REPS:
            for n_samples in self.N_SAMPLE_POINTS:
                data.append(dict(n_reps=n_reps, n_samples=n_samples))

        return self.generate_tests(data, marks=marks)

    def sample_shape_test_data(self):
        return self.generate_random_data_with_samples()

    def sample_belongs_to_support_test_data(self):
        return self.generate_random_data_with_samples()

    def point_to_pdf_vec_test_data(self):
        return self.generate_repeated_data_with_samples()

    def point_to_cdf_vec_test_data(self):
        return self.generate_repeated_data_with_samples()

    def point_to_cdf_bounds_test_data(self):
        return self.generate_random_data_with_samples()
