from polpo.testing.geometric import GeometricCaseData


class RetractionTestData(GeometricCaseData):
    def retraction_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"), on_metric=True
        )

    def lifting_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "point"),
            dependencies={"point": "base_point"},
            samplers={"point": "random_point_near"},
            on_metric=True,
        )

    def retraction_belongs_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def lifting_is_tangent_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "point"),
            dependencies={"point": "base_point"},
            samplers={"point": "random_point_near"},
        )

    def lifting_after_retraction_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def retraction_after_lifting_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "point"),
            dependencies={"point": "base_point"},
            samplers={"point": "random_point_near"},
        )
