from polpo.testing.geometric import FiberBundleCaseData
from polpo.testing.lazy import LazyValue


class FiberBundleTestData(FiberBundleCaseData):
    def riemannian_submersion_belongs_to_base_test_data(self):
        return self.generate_random_data(arg_names="point")

    def lift_belongs_to_total_space_test_data(self):
        return self.generate_random_data(
            arg_names="point",
            data_space="base",
        )

    def riemannian_submersion_after_lift_test_data(self):
        return self.generate_random_data(
            arg_names="point",
            data_space="base",
        )

    def log_after_align_is_horizontal_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "point"))

    def riemannian_submersion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("point"),
        )

    def lift_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("point"),
            data_space="base",
        )

    def align_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "point"),
        )

    def integrability_tensor_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b"),
        )

    def integrability_tensor_derivative_vec_test_data(self):
        datum = self._generate_lifted_random_datum(
            point_name="base_point",
            horizontal_names=("horizontal_vec_x", "horizontal_vec_y"),
            tangent_names=("nabla_x_y", "tangent_vec_e", "nabla_x_e"),
            n_points=1,
        )

        expected = LazyValue(
            lambda **kwargs: self.total_space.fiber_bundle.integrability_tensor_derivative(
                **kwargs
            ),
            **datum,
        )

        return self._vectorize_datum(
            datum,
            expected,
            vectorization_type="basic",
            expected_name=(
                "expected_nabla_x_a_y_e",
                "expected_a_y_e",
            ),
        )


class FiberBundleSubmersionLiftTestData(FiberBundleCaseData):
    """Test data for fiber bundles defined through submersion and horizontal lift.

    This test suite targets implementations that provide
    `tangent_riemannian_submersion` and `horizontal_lift`, and rely on the
    generic `FiberBundle` implementation for horizontal and vertical
    projections.

    It checks the two fundamental properties of this implementation:

    - `tangent_riemannian_submersion` maps tangent vectors of the total
      space to tangent vectors of the base space;
    - `horizontal_lift` is a right inverse of
      `tangent_riemannian_submersion`.

    These properties are sufficient to derive the corresponding horizontal
    and vertical projection identities, since `horizontal_projection` is
    obtained by composing `horizontal_lift` with
    `tangent_riemannian_submersion`, and `vertical_projection` is its
    complement.
    """

    def tangent_riemannian_submersion_is_tangent_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def tangent_riemannian_submersion_after_horizontal_lift_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
            data_space="base",
        )

    def tangent_riemannian_submersion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def horizontal_lift_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
            data_space="base",
        )


class FiberBundleProjectionTestData(FiberBundleCaseData):
    def horizontal_projection_is_horizontal_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def vertical_projection_is_vertical_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def tangent_riemannian_submersion_after_vertical_projection_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def horizontal_lift_is_horizontal_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
            data_space="base",
        )

    def horizontal_projection_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def vertical_projection_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def is_horizontal_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def is_vertical_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
        )


class GeneralLinearBuresWassersteinBundleTestData(FiberBundleTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {
        "align_vec": {"atol": 1e-3},
        "log_after_align_is_horizontal": {"atol": 1e-3},
    }
