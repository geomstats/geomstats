from geomstats.test.data import TestData

FROM_COORDS = ["ball", "half-space", "extrinsic"]
TO_COORDS = FROM_COORDS + ["intrinsic"]


class HyperbolicTestData(TestData):
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
