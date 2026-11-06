from polpo.testing.geometric import GeometricCaseData

from .manifold import _ManifoldMixinsTestData


class ComplexManifoldTestData(_ManifoldMixinsTestData, GeometricCaseData):
    def random_point_is_complex_test_data(self):
        return self.generate_random_data(arg_names="point")

    def random_point_imaginary_nonzero_test_data(self):
        return self.generate_random_data(arg_names="point")

    def random_tangent_vec_is_complex_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def random_tangent_vec_imaginary_nonzero_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))
