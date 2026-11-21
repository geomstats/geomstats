from polpo.testing.geometric import GeometricCaseData

from .nfold_manifold import NFoldManifoldTestData


class DiscreteCurvesStartingAtOriginTestData(NFoldManifoldTestData):
    def interpolate_vec_test_data(self):
        return self.generate_timed_vectorization_data(
            arg_names="point",
            op_name="interpolate",
        )

    def length_vec_test_data(self):
        return self.generate_vectorization_data(arg_names="point")

    def normalize_vec_test_data(self):
        return self.generate_vectorization_data(arg_names="point")

    def interpolate_at_sampling_points_test_data(self):
        return self.generate_random_data(arg_names="point")

    def normalize_is_unit_length_test_data(self):
        return self.generate_random_data(arg_names="point")


class ElasticMetricDistanceTestData(GeometricCaseData):
    def dist_against_no_transform_test_data(self):
        return self.generate_random_data(arg_names=("point_a", "point_b"))
