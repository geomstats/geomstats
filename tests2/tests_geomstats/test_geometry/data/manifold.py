from geomstats.test.data import TestData


class _ManifoldMixinsTestData:
    def belongs_vec_test_data(self):
        return self.generate_vec_data()

    def not_belongs_test_data(self):
        return self.generate_random_data()

    def random_point_belongs_test_data(self):
        return self.generate_random_data()

    def random_point_shape_test_data(self):
        return self.generate_shape_data()

    def is_tangent_vec_test_data(self):
        return self.generate_vec_data()

    def to_tangent_vec_test_data(self):
        return self.generate_vec_data()

    def to_tangent_is_tangent_test_data(self):
        return self.generate_random_data()

    def regularize_belongs_test_data(self):
        return self.generate_random_data()

    def regularize_vec_test_data(self):
        return self.generate_vec_data()

    def random_tangent_vec_is_tangent_test_data(self):
        return self.generate_random_data()

    def random_tangent_vec_shape_test_data(self):
        return self.generate_shape_data()


class ManifoldTestData(_ManifoldMixinsTestData, TestData):
    pass
