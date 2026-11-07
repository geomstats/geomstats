from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.information_geometry.gamma import (
    GammaDistributions,
    GammaDistributionsRandomVariable,
    NaturalToStandardDiffeo,
)
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.information_geometry.gamma import (
    GammaDistributionsTestCase,
    GammaMetricTestCase,
    StandardGammaDistributions,
)

from .data.gamma import (
    GammaDistributionsTestData,
    GammaMetricTestData,
    NaturalToStandardDiffeoTestData,
)


class TestNaturalToStandardDiffeo(
    DiffeoTestCase, metaclass=GeometricDataBasedParametrizer
):
    domain_space = GammaDistributions(equip=False)
    image_space = StandardGammaDistributions()
    diffeo = NaturalToStandardDiffeo()

    testing_data = NaturalToStandardDiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestGammaDistributions(
    GammaDistributionsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = GammaDistributions(equip=False)
    random_variable = GammaDistributionsRandomVariable(space)
    testing_data = GammaDistributionsTestData(space)


class TestGammaMetric(GammaMetricTestCase, metaclass=GeometricDataBasedParametrizer):
    space = GammaDistributions()
    testing_data = GammaMetricTestData(
        space,
        excluded_methods=(
            "injectivity_radius",
            "parallel_transport_ivp",
            "parallel_transport_bvp",
            "parallel_transport_ivp_transported_is_tangent",
            "parallel_transport_bvp_transported_is_tangent",
            "parallel_transport_ivp_norm",
            "parallel_transport_bvp_norm",
        ),
        method_marks={
            "xfail": (
                "dist_is_symmetric",
                "squared_dist_is_symmetric",
                "log_after_exp",
                "exp_after_log",
            )
        },
        generator_kwargs={"amplitude": 20.0},
    )
