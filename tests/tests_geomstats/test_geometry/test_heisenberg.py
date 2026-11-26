import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.heisenberg import HeisenbergVectors, HeisenbergVectorsToMatrices
from geomstats.test_cases.geometry.base import VectorSpaceTestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.lie_group import LieGroupTestCase

from .data.diffeo import DiffeoTestData
from .data.heisenberg import (
    HeisenbergVectorsTestData,
    HeisenbergVectorsToMatricesSmokeTestData,
)


class TestHeisenbergVectorsToMatrices(
    DiffeoTestCase, metaclass=GeometricDataBasedParametrizer
):
    domain_space = HeisenbergVectors(equip=False)
    diffeo = HeisenbergVectorsToMatrices()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        map_=diffeo,
        # TODO: unexclude once the image space is implemented
        excluded_methods=(
            "inverse",
            "inverse_tangent",
            "diffeomorphism_belongs",
            "inverse_belongs",
            "diffeomorphism_after_inverse",
            "tangent_is_tangent",
            "inverse_tangent_is_tangent",
            "tangent_after_inverse_tangent",
            "inverse_tangent_with_base_point",
        ),
    )


@pytest.mark.smoke
class TestHeisenbergVectorsToMatricesSmoke(
    DiffeoTestCase, metaclass=DataBasedParametrizer
):
    diffeo = HeisenbergVectorsToMatrices()
    testing_data = HeisenbergVectorsToMatricesSmokeTestData()


class TestHeisenbergVectors(
    LieGroupTestCase, VectorSpaceTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = HeisenbergVectors(equip=False)
    testing_data = HeisenbergVectorsTestData(
        space,
        excluded_methods=(
            "lie_bracket",
            "lie_bracket_belongs",
            "lie_bracket_antisymmetry",
            "lie_bracket_jacobi_identity",
            "lie_bracket_bilinearity",
        ),
    )
