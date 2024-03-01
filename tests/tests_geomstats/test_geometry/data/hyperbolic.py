import geomstats.backend as gs
from geomstats.test.data import TestData

from .riemannian_metric import (
    RiemannianMetricCmpWithTransformTestData,
    RiemannianMetricTestData,
)

FROM_COORDS = ["ball", "half-space", "extrinsic"]
TO_COORDS = FROM_COORDS + ["intrinsic"]
TO_COORDS_VEC = FROM_COORDS


class HyperbolicCoordsTransformTestData(TestData):
    def half_space_to_ball_tangent_vec_test_data(self):
        return self.generate_vec_data()

    def half_space_to_ball_tangent_is_tangent_test_data(self):
        return self.generate_random_data()

    def ball_to_half_space_tangent_vec_test_data(self):
        return self.generate_vec_data()

    def ball_to_half_space_tangent_is_tangent_test_data(self):
        return self.generate_random_data()

    def change_coordinates_system_vec_test_data(self):
        data = []
        for from_ in FROM_COORDS:
            for to in TO_COORDS:
                for n_reps in self.N_VEC_REPS:
                    data.append(
                        dict(
                            n_reps=n_reps,
                            from_coordinates_system=from_,
                            to_coordinates_system=to,
                        )
                    )
        return self.generate_tests(data)

    def change_coordinates_system_after_change_coordinates_system_test_data(self):
        data = []
        for from_ in FROM_COORDS:
            for to in TO_COORDS:
                for n_points in self.N_RANDOM_POINTS:
                    data.append(
                        dict(
                            n_points=n_points,
                            from_coordinates_system=from_,
                            to_coordinates_system=to,
                        )
                    )
        return self.generate_tests(data)

    # def change_tangent_coordinates_system_vec_test_data(self):
    #     data = []
    #     for from_ in FROM_COORDS:
    #         for to in TO_COORDS_VEC:
    #             for n_reps in self.N_VEC_REPS:
    #                 data.append(
    #                     dict(
    #                         n_reps=n_reps,
    #                         from_coordinates_system=from_,
    #                         to_coordinates_system=to,
    #                     )
    #                 )
    #     return self.generate_tests(data)


class HyperbolicCoordsTransform2TestData(TestData):
    def change_coordinates_system_test_data(self):
        data = [
            dict(
                from_coordinates_system="half-space",
                to_coordinates_system="ball",
                point=gs.array([0.0, 1.0]),
                expected=gs.zeros(2),
            ),
            dict(
                from_coordinates_system="half-space",
                to_coordinates_system="ball",
                point=gs.array([[0.0, 1.0], [0.0, 2.0]]),
                expected=gs.array([[0.0, 0.0], [0.0, 1.0 / 3.0]]),
            ),
            dict(
                from_coordinates_system="extrinsic",
                to_coordinates_system="ball",
                point=gs.array([2.0, 1.0, gs.sqrt(2)]),
                expected=gs.array([1.0 / 3.0, gs.sqrt(2) / 3.0]),
            ),
            dict(
                from_coordinates_system="extrinsic",
                to_coordinates_system="ball",
                point=gs.array([[2.0, gs.sqrt(2), 1.0], [2.0, 1.0, gs.sqrt(2)]]),
                expected=gs.array([[gs.sqrt(2) / 3.0, 1.0 / 3.0], [1.0 / 3.0, gs.sqrt(2) / 3.0]]),
            )
        ]
        return self.generate_tests(data)

    def change_tangent_coordinates_system_test_data(self):
        data = [
            dict(
                from_coordinates_system="extrinsic",
                to_coordinates_system="ball",
                tangent_vec=gs.array([1., 1., 1.]),
                base_point=gs.array([1.0, 0.0, 0.0]),
                expected=gs.array([1.0 / 2.0, 1.0 / 2.0]),
            ),
            dict(
                from_coordinates_system="extrinsic",
                to_coordinates_system="ball",
                tangent_vec=gs.array([[1., 2., 1.], [1., 1., 2.]]),
                base_point=gs.array([1.0, 0.0, 0.0]),
                expected=gs.array([[1.0, 1.0 / 2.0], [1.0 / 2.0, 1.0]]),
            ),
        ]
        return self.generate_tests(data)


class HyperbolicCmpWithTransformTestData(RiemannianMetricCmpWithTransformTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False


class HyperbolicMetricTestData(RiemannianMetricTestData):
    pass
