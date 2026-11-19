"""Information manifolds of normal distributions.

Lead authors: Antoine Collas, Alice Le Brigant.
"""

import geomstats.errors as errors
from geomstats.information_geometry.normal.centered import (
    CenteredNormalDistributions,
    CenteredNormalMetric,
    SharedMeanNormalDistributionsRandomVariable,
)
from geomstats.information_geometry.normal.diagonal import (
    DiagonalNormalDistributions,
    DiagonalNormalDistributionsRandomVariable,
    DiagonalNormalMetric,
)
from geomstats.information_geometry.normal.general import (
    GeneralNormalDistributions,
    MultivariateNormalDistributionsRandomVariable,
)
from geomstats.information_geometry.normal.univariate import (
    UnivariateNormalDistributions,
    UnivariateNormalDistributionsRandomVariable,
    UnivariateNormalMetric,
)


class NormalDistributions:
    """Class for the normal distributions.

    This class is a common interface to the following different situations:

    - univariate normal distributions
    - centered multivariate normal distributions
    - multivariate normal distributions with diagonal covariance matrix
    - general multivariate normal distributions

    Parameters
    ----------
    sample_dim : int
        Dimension of the sample space of the normal distribution.
    distribution_type : str, {'centered', 'diagonal', 'general'}
        Type of distributions.
        Optional, default: 'general'.
    """

    def __new__(cls, sample_dim, distribution_type="general", equip=True):
        """Instantiate class that corresponds to the distribution_type."""
        errors.check_parameter_accepted_values(
            distribution_type,
            "distribution_type",
            ["centered", "diagonal", "general"],
        )
        if sample_dim == 1:
            return UnivariateNormalDistributions(equip=equip)
        if distribution_type == "centered":
            return CenteredNormalDistributions(sample_dim, equip=equip)
        if distribution_type == "diagonal":
            return DiagonalNormalDistributions(sample_dim, equip=equip)
        return GeneralNormalDistributions(sample_dim, equip=equip)
