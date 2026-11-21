from polpo.testing.geometric import GeometricCaseData

from .mixins import DistMixinsTestData, GeodesicBVPMixinsTestData


class AlignQuotientMetricTestData(
    DistMixinsTestData, GeodesicBVPMixinsTestData, GeometricCaseData
):
    """Quotient metric data for bundles only implementing `align`.

    Distances and boundary value geodesics only need alignment, while the
    other operations need horizontal lifts or projections.
    """
