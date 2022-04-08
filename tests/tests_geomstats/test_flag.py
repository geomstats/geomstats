from geomstats.geometry.flag import Flag
from tests.conftest import Parametrizer
from tests.data_generation import _ManifoldTestData
from tests.geometry_test_cases import ManifoldTestCase


class TestFlag(ManifoldTestCase, metaclass=Parametrizer):
    space = Flag
    skip_test_random_point_belongs = True
    skip_test_projection_belongs = True
    skip_test_to_tangent_is_tangent_test_data = True
    skip_test_random_tangent_vec_is_tangent = True

    class TestDataFlag(_ManifoldTestData):

        def random_point_belongs_test_data(self):
            pass

        def projection_belongs_test_data(self):
            pass

        def to_tangent_is_tangent_test_data(self):
            pass

        def random_tangent_vec_is_tangent_test_data(self):
            pass

    testing_data = TestDataFlag()
