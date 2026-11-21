from ....test_geometry.data.base import VectorSpaceOpenSetTestData
from ..base import InformationManifoldMixinTestData


class CenteredNormalDistributionsTestData(
    InformationManifoldMixinTestData, VectorSpaceOpenSetTestData
):
    def point_to_pdf_against_scipy_test_data(self):
        return self._generate_random_data_with_samples()
