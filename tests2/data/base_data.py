import random

from geomstats.test.data import TestData


class _ProjectionMixinsTestData:
    def projection_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def projection_belongs_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)


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

    def is_tangent_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def to_tangent_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def to_tangent_is_tangent_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)


class VectorSpaceTestData(_ProjectionMixinsTestData, ManifoldTestData):
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


class MatrixLieAlgebraTestData(VectorSpaceTestData):
    def baker_campbell_hausdorff_vec_test_data(self):
        order = [2] + random.sample(range(3, 10), 1)
        data = []
        for order_ in order:
            data.extend(
                [dict(n_reps=n_reps, order=order_) for n_reps in self.N_VEC_REPS]
            )

        return self.generate_tests(data)

    def basis_representation_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def basis_representation_and_basis_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def matrix_representation_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def matrix_representation_belongs_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def matrix_representation_after_basis_representation_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def basis_representation_after_matrix_representation_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)


class LevelSetTestData(_ProjectionMixinsTestData, ManifoldTestData):
    def submersion_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def tangent_submersion_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)
