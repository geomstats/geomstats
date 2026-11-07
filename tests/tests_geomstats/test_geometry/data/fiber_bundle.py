from polpo.testing.geometric import FiberBundleCaseData, GeometricCaseData
from polpo.testing.lazy import LazyValue


class FiberBundleTestData(FiberBundleCaseData):
    def with_redundancy(
        self,
        projection_source=None,
        lift_uses_horizontal_projection=False,
        custom_is_horizontal=False,
    ):
        """Mark tests that are redundant given the implementation.

        Parameters
        ----------
        projection_source : {"vertical", "horizontal", "submersion_lift"} or None
            Independently implemented operation from which projections are
            derived. None makes no redundancy assumptions.
        lift_uses_horizontal_projection : bool
            Whether horizontal lift is computed through horizontal
            projection.
        custom_is_horizontal : bool
            Whether `is_horizontal` is overridden.

        Returns
        -------
        testing_data : FiberBundleTestData
            Self, so calls can be chained.

        Notes
        -----
        A name covers both a test and its vectorization test.

        1. If ``projection_source="vertical"``, then horizontal projection
           is the identity minus vertical projection, and its direct tests
           are redundant.
        2. If ``projection_source="horizontal"``, then vertical projection
           is the identity minus horizontal projection, and its direct
           tests are redundant.
        3. If ``projection_source="submersion_lift"``, then both
           projections are derived from horizontal lift and tangent
           submersion, and their direct tests are redundant. The image of
           vertical projection lies in the kernel of the tangent
           submersion by the right-inverse property of horizontal lift,
           so `vertical_projection_is_vertical` is also redundant.
        4. If a projection source is given and `is_horizontal` is not
           overridden, then `horizontal_projection_is_horizontal` is
           implied by `projections_are_orthogonal`.
        5. If horizontal lift is computed through horizontal projection,
           then its direct tests are redundant.
        """
        redundant = set()

        if projection_source == "vertical":
            redundant.add("horizontal_projection")
        elif projection_source == "horizontal":
            redundant.add("vertical_projection")
        elif projection_source == "submersion_lift":
            redundant |= {
                "horizontal_projection",
                "vertical_projection",
                "vertical_projection_is_vertical",
            }
        elif projection_source is not None:
            raise ValueError(f"Unknown projection source: {projection_source!r}.")

        if projection_source is not None and not custom_is_horizontal:
            redundant.add("horizontal_projection_is_horizontal")

        if lift_uses_horizontal_projection:
            redundant |= {
                "horizontal_lift",
                "horizontal_lift_is_horizontal",
                "horizontal_lift_with_base_point",
            }

        marked = set(self.method_marks.get("redundant", ()))
        self.method_marks = {**self.method_marks, "redundant": marked | redundant}

        return self

    def riemannian_submersion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("point"),
        )

    def lift_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("point"),
            data_space="base",
        )

    def tangent_riemannian_submersion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def align_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "point"),
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

    def horizontal_lift_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
            data_space="base",
        )

    def integrability_tensor_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec_x", "tangent_vec_e"),
        )

    def integrability_tensor_derivative_vec_test_data(self):
        datum = self._generate_lifted_random_datum(
            point_name="base_point",
            horizontal_names=("horizontal_vec_x", "horizontal_vec_y"),
            tangent_names=("nabla_x_y", "tangent_vec_e", "nabla_x_e"),
            n_points=1,
        )

        def _nabla_x_y(horizontal_vec_x, horizontal_vec_y, tangent_vec, base_point):
            fiber_bundle = self.total_space.fiber_bundle
            return fiber_bundle.integrability_tensor(
                horizontal_vec_x, horizontal_vec_y, base_point
            ) + fiber_bundle.horizontal_projection(tangent_vec, base_point)

        datum["nabla_x_y"] = LazyValue(
            _nabla_x_y,
            datum["horizontal_vec_x"],
            datum["horizontal_vec_y"],
            datum["nabla_x_y"],
            datum["base_point"],
            label="n=1",
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

    def tangent_riemannian_submersion_is_tangent_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def log_after_align_is_horizontal_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "point"))

    def projections_are_orthogonal_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )

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

    def tangent_riemannian_submersion_after_horizontal_lift_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
            data_space="base",
        )

    def horizontal_lift_with_base_point_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
            data_space="base",
        )


class AlignInSameFiberTestData(GeometricCaseData):
    """Data to align a point of the fiber of a base point.

    `space` is the total space. Its data generator must implement
    `random_point_in_fiber`.
    """

    def align_in_same_fiber_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "point"),
            samplers={"point": "random_point_in_fiber"},
            dependencies={"point": "base_point"},
        )


class AlignComparisonTestData(GeometricCaseData):
    def align_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "point"))
