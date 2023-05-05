import geomstats.backend as gs
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import SPDBuresWassersteinMetric, SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.data_generation import TestData


class BuresWassersteinBundle(FiberBundle):
    def __init__(self, total_space):
        super().__init__(
            total_space=total_space,
            group=SpecialOrthogonal(total_space.n),
        )

    @staticmethod
    def default_metric():
        return MatricesMetric

    @staticmethod
    def riemannian_submersion(point):
        return Matrices.mul(point, Matrices.transpose(point))

    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        product = Matrices.mul(base_point, Matrices.transpose(tangent_vec))
        return 2 * Matrices.to_symmetric(product)

    def horizontal_lift(self, tangent_vec, base_point=None, fiber_point=None):
        if base_point is None:
            if fiber_point is not None:
                base_point = self.riemannian_submersion(fiber_point)
            else:
                raise ValueError(
                    "Either a point (of the total space) or a "
                    "base point (of the base manifold) must be "
                    "given."
                )
        sylvester = gs.linalg.solve_sylvester(base_point, base_point, tangent_vec)
        return Matrices.mul(sylvester, fiber_point)

    @staticmethod
    def lift(point):
        return gs.linalg.cholesky(point)


class BundleTestData(TestData):
    Bundle = BuresWassersteinBundle
    TotalSpace = GeneralLinear
    Base = SPDMatrices

    def riemannian_submersion_test_data(self):
        random_data = [dict(n=2)]
        return self.generate_tests([], random_data)

    def lift_and_riemannian_submersion_test_data(self):
        random_data = [dict(n=2)]
        return self.generate_tests([], random_data)

    def tangent_riemannian_submersion_test_data(self):
        random_data = [dict(n=2)]
        return self.generate_tests([], random_data)

    def horizontal_projection_test_data(self):
        random_data = [dict(n=2)]
        return self.generate_tests([], random_data)

    def vertical_projection_test_data(self):
        random_data = [dict(n=2)]
        return self.generate_tests([], random_data)

    def horizontal_lift_and_tangent_riemannian_submersion_test_data(self):
        random_data = [dict(n=2)]
        return self.generate_tests([], random_data)

    def is_horizontal_test_data(self):
        random_data = [dict(n=2)]
        return self.generate_tests([], random_data)

    def is_vertical_test_data(self):
        random_data = [dict(n=2)]
        return self.generate_tests([], random_data)

    def align_test_data(self):
        random_data = [dict(n=2)]
        return self.generate_tests([], random_data)


class QuotientMetricTestData(TestData):

    Base = SPDMatrices
    Bundle = BuresWassersteinBundle
    TotalSpace = GeneralLinear

    ReferenceMetric = SPDBuresWassersteinMetric
    Metric = QuotientMetric

    def inner_product_test_data(self):
        random_data = [dict(n=2)]
        return self.generate_tests([], random_data)

    def exp_test_data(self):
        random_data = [dict(n=2)]
        return self.generate_tests([], random_data)

    def log_test_data(self):
        random_data = [dict(n=2)]
        return self.generate_tests([], random_data)

    def squared_dist_test_data(self):
        random_data = [dict(n=2)]
        return self.generate_tests([], random_data)

    def integrability_tensor_test_data(self):
        random_data = [dict(n=2)]
        return self.generate_tests([], random_data)
