import random

from geomstats.test.data import TestData


class ManifoldTestData(TestData):
    N_VEC_REPS = random.sample(range(2, 5), 1)
    N_SHAPE_POINTS = [1] + random.sample(range(2, 5), 1)
    N_RANDOM_POINTS = [1] + random.sample(range(2, 5), 1)

    def belongs_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def not_belongs_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def random_point_belongs_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def random_point_shape_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_SHAPE_POINTS]
        return self.generate_tests(data)


class VectorSpaceTestData(ManifoldTestData):
    def projection_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def projection_belongs_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def is_tangent_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def to_tangent_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def to_tangent_is_tangent_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def basis_cardinality_test_data(self):
        return None

    def basis_belongs_test_data(self):
        return self.generate_tests([dict()])

    def random_point_is_tangent_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def to_tangent_is_projection_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)


class MatrixVectorSpaceMixinsTestData(TestData):
    def to_vector_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def to_vector_and_basis_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def from_vector_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def from_vector_belongs_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def from_vector_after_to_vector_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def to_vector_after_from_vector_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)
