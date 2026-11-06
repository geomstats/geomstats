from .manifold import ManifoldTestData


class ProductManifoldTestData(ManifoldTestData):
    def embed_to_product_after_project_from_product_test_data(self):
        return self.generate_random_data(arg_names="point")
