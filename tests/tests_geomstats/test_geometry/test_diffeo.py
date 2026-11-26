from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.test_case import autodiff_backend
from geomstats.test_cases.geometry.diffeo import AutodiffDiffeoTestCase, CircleSO2Diffeo

from .data.diffeo import AutodiffDiffeoTestData


class TestCircleSO2Diffeo(
    AutodiffDiffeoTestCase, metaclass=GeometricDataBasedParametrizer
):
    domain_space = Hypersphere(dim=1, equip=False)
    image_space = SpecialOrthogonal(n=2, point_type="matrix", equip=False)
    diffeo = CircleSO2Diffeo()

    testing_data = AutodiffDiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
        excluded_methods=()
        if autodiff_backend()
        else (
            "tangent",
            "inverse_tangent",
            "tangent_is_tangent",
            "inverse_tangent_is_tangent",
            "inverse_tangent_after_tangent",
            "tangent_after_inverse_tangent",
            "tangent_with_image_point",
            "inverse_tangent_with_base_point",
            "jacobian",
            "inverse_jacobian",
        ),
    )
