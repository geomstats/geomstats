from ....test_geometry.data.product_manifold import ProductManifoldTestData
from ..base import InformationManifoldMixinTestData


class GeneralNormalDistributionsTestData(
    InformationManifoldMixinTestData, ProductManifoldTestData
):
    def point_to_pdf_against_scipy_test_data(self):
        return self._generate_random_data_with_samples()
