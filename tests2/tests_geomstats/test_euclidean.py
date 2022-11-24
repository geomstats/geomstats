from geomstats.geometry.euclidean import Euclidean
from geomstats.test.geometry.euclidean import EuclideanTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.euclidean_data import EuclideanTestData


class TestEuclidean(EuclideanTestCase, metaclass=DataBasedParametrizer):
    space = Euclidean(2)
    testing_data = EuclideanTestData()
