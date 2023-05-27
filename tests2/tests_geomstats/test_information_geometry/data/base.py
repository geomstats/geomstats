import random

from geomstats.test.data import TestData


class InformationManifoldMixinTestData(TestData):
    N_SAMPLE_POINTS = [1] + random.sample(range(2, 5), 1)

    # TODO: uncomment
    # def sample_shape_test_data(self, marks=()):
    #     data = []
    #     for n_points in self.N_RANDOM_POINTS:
    #         for n_samples in self.N_SAMPLE_POINTS:
    #             data.append(dict(n_points=n_points, n_samples=n_samples))

    #     return self.generate_tests(data, marks=marks)

    # TODO: uncomment
    # def point_to_pdf_vec_test_data(self):
    #     return self.generate_vec_data()

    # def point_to_cdf_vec_test_data(self):
    #     return self.generate_vec_data()
