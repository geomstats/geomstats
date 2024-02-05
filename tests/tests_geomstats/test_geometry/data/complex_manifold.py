from geomstats.test.data import TestData

from .manifold import _ManifoldMixinsTestData


class ComplexManifoldTestData(_ManifoldMixinsTestData, TestData):
    def random_point_is_complex_test_data(self):
        return self.generate_random_data()

    def random_point_imaginary_nonzero_test_data(self):
        return self.generate_tests([dict(n_points=self.N_RANDOM_POINTS[-1])])

    def random_tangent_vec_is_complex_test_data(self):
        return self.generate_random_data()

    def random_tangent_vec_imaginary_nonzero_test_data(self):
        return self.generate_tests([dict(n_points=self.N_RANDOM_POINTS[-1])])
