from geomstats.test.data import TestData

from .manifold import _ManifoldMixinsTestData


class ComplexManifoldTestData(_ManifoldMixinsTestData, TestData):
    def random_point_is_complex_test_data(self):
        return self.generate_random_data()

    def random_point_imaginary_nonzero_test_data(self):
        return self.generate_tests([dict(n_points=5)])
