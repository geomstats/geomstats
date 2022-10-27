"""Statistical Manifold of categorical distributions with the Fisher metric.

Lead author: Alice Le Brigant.
"""
from geomstats.information_geometry.multinomial import (
    MultinomialDistributions,
    MultinomialMetric,
)


class CategoricalDistributions(MultinomialDistributions):
    r"""Class for the manifold of categorical distributions.

    This is the set of `n+1`-tuples of positive reals that sum up to one,
    i.e. the `n`-simplex. Each point is the parameter of a categorical
    distribution, i.e. gives the probabilities of $n$ different outcomes
    in a single experiment.

    Attributes
    ----------
    dim : int
        Dimension of the manifold of categorical distributions. The
        number of outcomes is dim + 1.
    embedding_manifold : Manifold
        Embedding manifold.
    """

    def __init__(self, dim, **kwargs):
        kwargs.setdefault("metric", CategoricalMetric(dim=dim))
        super().__init__(
            dim=dim,
            n_draws=1,
            **kwargs
        )


class CategoricalMetric(MultinomialMetric):
    """Class for the Fisher information metric on categorical distributions.

    The Fisher information metric on the $n$-simplex of categorical
    distributions parameters can be obtained as the pullback metric of the
    $n$-sphere using the componentwise square root.

    References
    ----------
    .. [K2003] R. E. Kass. The Geometry of Asymptotic Inference. Statistical
        Science, 4(3): 188 - 234, 1989.
    """

    def __init__(self, dim):
        super().__init__(dim=dim, n_draws=1)
