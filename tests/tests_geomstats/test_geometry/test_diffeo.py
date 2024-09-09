from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.diffeo import AutodiffDiffeoTestCase, CircleSO2Diffeo

from .data.diffeo import AutodiffDiffeoTestData


class TestCircleSO2Diffeo(AutodiffDiffeoTestCase, metaclass=DataBasedParametrizer):
    space = Hypersphere(dim=1, equip=False)
    image_space = SpecialOrthogonal(n=2, point_type="matrix", equip=False)
    diffeo = CircleSO2Diffeo()
    testing_data = AutodiffDiffeoTestData()
