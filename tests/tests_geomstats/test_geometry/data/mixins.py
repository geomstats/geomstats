class ProjectionMixinsTestData:
    def projection_vec_test_data(self):
        return self.generate_vec_data()

    def projection_belongs_test_data(self):
        return self.generate_random_data()


class GroupExpMixinsTestData:
    def exp_vec_test_data(self):
        return self.generate_vec_data()


class DistMixinsTestData:
    def dist_vec_test_data(self):
        return self.generate_vec_data()

    def dist_is_symmetric_test_data(self):
        return self.generate_random_data()

    def dist_is_positive_test_data(self):
        return self.generate_random_data()

    def dist_point_to_itself_is_zero_test_data(self):
        return self.generate_random_data()

    def dist_triangle_inequality_test_data(self):
        return self.generate_random_data()


class GeodesicBVPMixinsTestData:
    def geodesic_bvp_vec_test_data(self):
        return self.generate_vec_data_with_time()

    def geodesic_boundary_points_test_data(self):
        return self.generate_random_data()

    def geodesic_bvp_reverse_test_data(self):
        return self.generate_random_data_with_time()

    def geodesic_bvp_belongs_test_data(self):
        return self.generate_random_data_with_time()
