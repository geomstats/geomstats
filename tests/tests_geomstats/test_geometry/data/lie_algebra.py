import random

from .base import MatrixVectorSpaceTestData


class MatrixLieAlgebraTestData(MatrixVectorSpaceTestData):
    def baker_campbell_hausdorff_vec_test_data(self):
        data = []
        for order in [2] + random.sample(range(3, 10), 1):
            data.extend(
                self.generate_vectorization_data(
                    arg_names=("matrix_a", "matrix_b"),
                    samplers={"matrix_a": "random_point", "matrix_b": "random_point"},
                    op_evaluator=lambda op, order=order, **kwargs: op(
                        **kwargs, order=order
                    ),
                    order=order,
                )
            )
        return data
