from tests2.tests_geomstats.test_geometry.data.manifold import ManifoldTestData
from tests2.tests_geomstats.test_geometry.data.riemannian_metric import (
    RiemannianMetricTestData,
)


class ProductManifoldTestData(ManifoldTestData):
    skips = ("not_belongs",)

    def embed_to_product_after_project_from_product_test_data(self):
        return self.generate_random_data()


class ProductRiemannianMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    trials = 3

    skips = (
        # TODO: remove skip
        "metric_matrix_is_spd",
        "inner_coproduct_vec",
        "inner_product_derivative_matrix_vec",
        "christoffels_vec",
    )
