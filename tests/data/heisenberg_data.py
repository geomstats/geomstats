import random

from geomstats.geometry.heisenberg import HeisenbergVectors
from tests.data_generation import _LieGroupTestData, _VectorSpaceTestData


class HeisenbergVectorsTestData(_LieGroupTestData, _VectorSpaceTestData):
    Space = HeisenbergVectors

    space_args_list = [()] * 3
    shape_list = [(3,)] * 3
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)
    n_tangent_vecs_list = random.sample(range(2, 5), 2)
    batch_shape_list = [
        tuple(random.choices(range(2, 10), k=i)) for i in random.sample(range(1, 5), 3)
    ]

    def dimension_test_data(self):
        smoke_data = [dict(expected=3)]
        return self.generate_tests(smoke_data)

    def belongs_test_data(self):
        smoke_data = [
            dict(point=[1.0, 2.0, 3.0, 4], expected=False),
            dict(
                point=[[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]],
                expected=[False, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def is_tangent_test_data(self):
        smoke_data = [
            dict(
                vector=[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
                expected=[False, False],
            )
        ]
        return self.generate_tests(smoke_data)

    def jacobian_translation_test_data(self):
        smoke_data = [
            dict(
                vec=[[1.0, -10.0, 0.2], [-2.0, 100.0, 0.5]],
                expected=[
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 0.5, 1.0]],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-50.0, -1.0, 1.0]],
                ],
            )
        ]
        return self.generate_tests(smoke_data)
