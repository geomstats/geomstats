from polpo.testing.geometric import GeometricCaseData

import geomstats.backend as gs


class _ManifoldMixinsTestData:
    def belongs_is_true_test_data(self):
        return self.generate_random_data(
            arg_names="point",
            atol=gs.atol,
        )

    def assert_is_tangent_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
            atol=gs.atol,
        )

    def regularize_belongs_test_data(self):
        return self.generate_random_data(
            arg_names="point",
            atol=gs.atol,
        )

    def belongs_vec_test_data(self):
        return self.generate_vectorization_data(arg_names="point")

    def regularize_vec_test_data(self):
        return self.generate_vectorization_data(arg_names="point")

    def is_tangent_vec_test_data(self):
        return self.generate_vectorization_data(arg_names=("base_point", "vector"))


class ManifoldTestData(_ManifoldMixinsTestData, GeometricCaseData):
    pass
