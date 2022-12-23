import random

import pytest

from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.geometry.special_orthogonal import (
    SpecialOrthogonal2VectorsTestCase,
    SpecialOrthogonal3VectorsTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.special_orthogonal_data import (
    SpecialOrthogonal2VectorsTestData,
    SpecialOrthogonal3VectorsTestData,
)

# @pytest.fixture(
#     scope="class",
#     params=[
#         2,
#         # 3,
#     ],
# )
# def spaces(request):
#     request.cls.space = SpecialOrthogonal(request.param, point_type="vector")


# @pytest.mark.usefixtures("spaces")
# class TestSpecialOrthogonalVectors(
#     SpecialOrthogonalVectorsTestCase, metaclass=DataBasedParametrizer
# ):
#     testing_data = SpecialOrthogonalVectorsTestData()


class TestSpecialOrthogonal2Vectors(
    SpecialOrthogonal2VectorsTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialOrthogonal(2, point_type="vector")
    testing_data = SpecialOrthogonal2VectorsTestData()


class TestSpecialOrthogonal3Vectors(
    SpecialOrthogonal3VectorsTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialOrthogonal(3, point_type="vector")
    testing_data = SpecialOrthogonal3VectorsTestData()
