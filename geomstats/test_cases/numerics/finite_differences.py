import geomstats.backend as gs
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase


class FiniteDifferenceTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    def _linear_path(self, base_point, tangent_vec, time):
        ijk = "ijk"[: self.space.point_ndim]
        if base_point.shape[: -self.space.point_ndim]:
            base_point = gs.expand_dims(base_point, axis=-(self.space.point_ndim + 1))

        return base_point + gs.einsum(f"t,...{ijk}->...t{ijk}", time, tangent_vec)
