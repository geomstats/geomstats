import geomstats.backend as gs
from geomstats.test.data import TestData
from tests2.data.base_data import VectorSpaceTestData


class SymmetricMatricesTestData(VectorSpaceTestData):
    def to_vector_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def from_vector_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def to_vector_and_basis_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def from_vector_after_to_vector_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def to_vector_after_from_vector_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)


class SymmetricMatrices1TestData(TestData):
    def basis_test_data(self):
        smoke_data = [
            dict(expected=gs.array([[[1.0]]])),
        ]
        return self.generate_tests(smoke_data)

    def to_vector_test_data(self):
        data = [dict(point=gs.array([[1.0]]), expected=gs.array([1.0]))]

        return self.generate_tests(data)

    def from_vector_test_data(self):
        data = [
            dict(vec=gs.array([1.0]), expected=gs.array([[1.0]])),
        ]

        return self.generate_tests(data)


class SymmetricMatrices2TestData(TestData):
    def basis_test_data(self):
        data = [
            dict(
                expected=gs.array(
                    [
                        [[1.0, 0.0], [0, 0]],
                        [[0, 1.0], [1.0, 0]],
                        [[0, 0.0], [0, 1.0]],
                    ]
                ),
            ),
        ]

        return self.generate_tests(data)

    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[1.0, 2.0], [2.0, 1.0]]), expected=gs.array(True)),
            dict(point=gs.array([[1.0, 1.0], [2.0, 1.0]]), expected=gs.array(False)),
            dict(
                point=gs.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [0.0, 1.0]]]),
                expected=gs.array([True, False]),
            ),
        ]
        return self.generate_tests(data)


class SymmetricMatrices3TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(
                point=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
                expected=gs.array(True),
            ),
        ]

        return self.generate_tests(data)

    def to_vector_test_data(self):
        data = [
            dict(
                point=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
                expected=gs.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
            dict(
                point=gs.array(
                    [
                        [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                        [[7.0, 8.0, 9.0], [8.0, 10.0, 11.0], [9.0, 11.0, 12.0]],
                    ]
                ),
                expected=gs.array(
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    ]
                ),
            ),
        ]
        return self.generate_tests(data)

    def from_vector_test_data(self):
        data = [
            dict(
                vec=gs.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                expected=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
            ),
            dict(
                vec=gs.array(
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    ]
                ),
                expected=gs.array(
                    [
                        [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                        [[7.0, 8.0, 9.0], [8.0, 10.0, 11.0], [9.0, 11.0, 12.0]],
                    ]
                ),
            ),
        ]
        return self.generate_tests(data)


class SymmetricMatricesOpsTestData(TestData):
    def expm_test_data(self):
        data = [
            dict(
                mat=gs.array([[0.0, 0.0], [0.0, 0.0]]),
                expected=gs.array([[1.0, 0.0], [0.0, 1.0]]),
            )
        ]
        return self.generate_tests(data)

    def powerm_test_data(self):
        data = [
            dict(
                mat=gs.array([[1.0, 2.0], [2.0, 3.0]]),
                power=1.0,
                expected=gs.array([[1.0, 2.0], [2.0, 3.0]]),
            ),
            dict(
                mat=gs.array([[1.0, 2.0], [2.0, 3.0]]),
                power=2.0,
                expected=gs.array([[5.0, 8.0], [8.0, 13.0]]),
            ),
        ]
        return self.generate_tests(data)
