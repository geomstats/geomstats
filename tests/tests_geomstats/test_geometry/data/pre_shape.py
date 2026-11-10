import random

from polpo.testing.geometric import GeometricCaseData
from polpo.testing.lazy import LazyValue

from geomstats.test.random import KendalShapeRandomDataGenerator

from .base import LevelSetTestData
from .fiber_bundle import FiberBundleTestData
from .riemannian_metric import RiemannianMetricTestData


def _vecs_kwargs(horizontal_names=(), vertical_names=(), tangent_names=()):
    """Arguments to sample tangent vectors at a random base point.

    Parameters
    ----------
    horizontal_names : tuple of str
        Names of the horizontal tangent vectors.
    vertical_names : tuple of str
        Names of the vertical tangent vectors.
    tangent_names : tuple of str
        Names of the tangent vectors.

    Returns
    -------
    kwargs : dict
        Arguments for `generate_random_data` or `generate_vectorization_data`.
    """
    samplers = {}
    for sampler, names in (
        ("random_horizontal_vec", horizontal_names),
        ("random_vertical_vec", vertical_names),
        ("random_tangent_vec", tangent_names),
    ):
        samplers.update(dict.fromkeys(names, sampler))

    return dict(
        arg_names=("base_point", *samplers),
        samplers=samplers,
        dependencies=dict.fromkeys(samplers, "base_point"),
    )


class PreShapeSpaceTestData(LevelSetTestData):
    def is_centered_vec_test_data(self):
        return self.generate_vectorization_data(arg_names="point")

    def center_vec_test_data(self):
        return self.generate_vectorization_data(arg_names="point")

    def center_is_centered_test_data(self):
        return self.generate_random_data(arg_names="point")


class PreShapeBundleTestData(FiberBundleTestData):
    def integrability_tensor_derivative_parallel_vec_test_data(self):
        return self.generate_vectorization_data(
            **_vecs_kwargs(
                horizontal_names=(
                    "horizontal_vec_x",
                    "horizontal_vec_y",
                    "horizontal_vec_z",
                )
            ),
            expected_name=("expected_nabla_x_a_y_z", "expected_a_y_z"),
            # NB: a_y_z does not depend on horizontal_vec_x
            vectorization_type="basic",
        )

    def iterated_integrability_tensor_derivative_parallel_vec_test_data(self):
        return self.generate_vectorization_data(
            **_vecs_kwargs(horizontal_names=("horizontal_vec_x", "horizontal_vec_y")),
            expected_name=(
                "expected_nabla_x_a_y_v",
                "expected_a_x_a_y_a_x_y",
                "expected_nabla_x_v",
                "expected_a_y_a_x_y",
                "expected_vertical_vec_v",
            ),
        )

    def vertical_projection_correctness_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def horizontal_projection_correctness_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def horizontal_projection_is_tangent_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def alignment_is_symmetric_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "point"))

    def integrability_tensor_identity_1_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )

    def integrability_tensor_identity_2_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )

    def integrability_tensor_alt_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )

    def integrability_tensor_derivative_is_alternate_test_data(self):
        return self.generate_random_data(
            **_vecs_kwargs(
                horizontal_names=(
                    "horizontal_vec_x",
                    "horizontal_vec_y",
                    "horizontal_vec_z",
                    "horizontal_vec_dy",
                    "horizontal_vec_dz",
                )
            )
        )

    def integrability_tensor_derivative_is_skew_symmetric_test_data(self):
        return self.generate_random_data(
            **_vecs_kwargs(
                horizontal_names=(
                    "horizontal_vec_x",
                    "horizontal_vec_y",
                    "horizontal_vec_z",
                    "horizontal_vec_dy",
                    "horizontal_vec_dz",
                ),
                tangent_names=("tangent_vec_v", "tangent_vec_dv"),
            )
        )

    def integrability_tensor_derivative_reverses_hor_test_data(self):
        return self.generate_random_data(
            **_vecs_kwargs(
                horizontal_names=(
                    "horizontal_vec_x",
                    "horizontal_vec_y",
                    "horizontal_vec_z",
                    "horizontal_vec_h",
                    "horizontal_vec_dy",
                    "horizontal_vec_dz",
                    "horizontal_vec_dh",
                )
            )
        )

    def integrability_tensor_derivative_reverses_ver_test_data(self):
        return self.generate_random_data(
            **_vecs_kwargs(
                horizontal_names=(
                    "horizontal_vec_x",
                    "horizontal_vec_y",
                    "horizontal_vec_dy",
                ),
                vertical_names=(
                    "vertical_vec_z",
                    "vertical_vec_w",
                    "vertical_vec_dz",
                    "vertical_vec_dw",
                ),
            )
        )

    def integrability_tensor_derivative_parallel_optimized_test_data(self):
        return self.generate_random_data(
            **_vecs_kwargs(
                horizontal_names=(
                    "horizontal_vec_x",
                    "horizontal_vec_y",
                    "horizontal_vec_z",
                )
            )
        )

    def iterated_integrability_tensor_derivative_parallel_optimized_test_data(self):
        return self.generate_random_data(
            **_vecs_kwargs(horizontal_names=("horizontal_vec_x", "horizontal_vec_y"))
        )


class KendallShapeMetricTestData(RiemannianMetricTestData):
    """Kendall shape metric data.

    Points and tangent vectors are sampled on `total_space`, where
    horizontal vectors are defined.
    """

    def __init__(self, space=None, total_space=None, **kwargs):
        super().__init__(space, **kwargs)
        self.total_space = total_space
        self.data_generator = LazyValue(
            lambda: KendalShapeRandomDataGenerator(
                self.total_space, **self.generator_kwargs
            )
        )

    def curvature_is_skew_operator_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )

    def curvature_bianchi_identity_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b", "tangent_vec_c")
        )

    def sectional_curvature_lower_bound_test_data(self):
        return self.generate_random_data(
            **_vecs_kwargs(horizontal_names=("horizontal_vec_a", "horizontal_vec_b"))
        )

    def curvature_derivative_bianchi_identity_test_data(self):
        return self.generate_random_data(
            **_vecs_kwargs(
                horizontal_names=(
                    "horizontal_vec_x",
                    "horizontal_vec_y",
                    "horizontal_vec_z",
                    "horizontal_vec_h",
                )
            )
        )

    def curvature_derivative_is_skew_operator_test_data(self):
        return self.generate_random_data(
            **_vecs_kwargs(
                horizontal_names=(
                    "horizontal_vec_x",
                    "horizontal_vec_y",
                    "horizontal_vec_z",
                )
            )
        )

    def directional_curvature_derivative_is_quadratic_test_data(self):
        return self.generate_random_data(
            **_vecs_kwargs(horizontal_names=("horizontal_vec_x", "horizontal_vec_y")),
            coef_x=random.uniform(0.0, 1.0),
            coef_y=random.uniform(0.0, 1.0),
        )

    def parallel_transport_ivp_transported_is_horizontal_test_data(self):
        return self.generate_random_data(
            **_vecs_kwargs(
                horizontal_names=("horizontal_vec",), tangent_names=("direction",)
            )
        )


class KendallShapeMetricCmpTestData(GeometricCaseData):
    def directional_curvature_derivative_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )
