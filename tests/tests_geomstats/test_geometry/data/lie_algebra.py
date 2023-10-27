import random

from .base import VectorSpaceTestData


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
        return self.generate_vec_data()

    def basis_representation_and_basis_test_data(self):
        return self.generate_random_data()

    def matrix_representation_vec_test_data(self):
        return self.generate_vec_data()

    def matrix_representation_belongs_test_data(self):
        return self.generate_random_data()

    def matrix_representation_after_basis_representation_test_data(self):
        return self.generate_random_data()

    def basis_representation_after_matrix_representation_test_data(self):
        return self.generate_random_data()
