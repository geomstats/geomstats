import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import VectorSpaceTestData
from .diffeo import DiffeoTestData
from .lie_group import LieGroupTestData


class HeisenbergVectorsToMatricesTestData(DiffeoTestData):
    # TODO: unskip once the image space is implemented
    skips = (
        "diffeomorphism_belongs",
        "inverse_vec",
        "inverse_belongs",
        "diffeomorphism_after_inverse",
        "tangent_is_tangent",
        "inverse_tangent_vec",
        "inverse_tangent_is_tangent",
        "inverse_tangent_with_base_point",
        "tangent_after_inverse_tangent",
    )


class HeisenbergVectorsToMatricesSmokeTestData(TestData):
    def diffeomorphism_test_data(self):
        data = [
            dict(
                base_point=gs.array([1.0, 2.0, 3.0]),
                expected=gs.array([[1.0, 1.0, 4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]),
            )
        ]
        return self.generate_tests(data)

    def inverse_test_data(self):
        data = [
            dict(
                image_point=gs.array(
                    [[1.0, 1.0, 4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]
                ),
                expected=gs.array([1.0, 2.0, 3.0]),
            )
        ]
        return self.generate_tests(data)

    def tangent_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([1.0, 0.0, 0.0]),
                base_point=gs.array([1.0, 2.0, 3.0]),
                expected=gs.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            )
        ]
        return self.generate_tests(data)

    def inverse_tangent_test_data(self):
        data = [
            dict(
                image_tangent_vec=gs.array(
                    [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                image_point=gs.array(
                    [[1.0, 1.0, 4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]
                ),
                expected=gs.array([1.0, 0.0, 0.0]),
            )
        ]
        return self.generate_tests(data)


class HeisenbergVectorsTestData(LieGroupTestData, VectorSpaceTestData):
    fail_for_not_implemented_errors = False


class HeisenbergVectors3TestData(TestData):
    def dim_test_data(self):
        smoke_data = [dict(expected=3)]
        return self.generate_tests(smoke_data)

    def belongs_test_data(self):
        data = [
            dict(point=gs.array([1.0, 2.0, 3.0, 4]), expected=False),
            dict(
                point=gs.array([[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]]),
                expected=[False, False],
            ),
        ]
        return self.generate_tests(data)

    def is_tangent_test_data(self):
        data = [
            dict(
                vector=gs.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
                base_point=None,
                expected=[False, False],
            )
        ]
        return self.generate_tests(data)

    def jacobian_translation_test_data(self):
        data = [
            dict(
                point=gs.array([[1.0, -10.0, 0.2], [-2.0, 100.0, 0.5]]),
                left=True,
                expected=gs.array(
                    [
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 0.5, 1.0]],
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-50.0, -1.0, 1.0]],
                    ]
                ),
            )
        ]
        return self.generate_tests(data)
