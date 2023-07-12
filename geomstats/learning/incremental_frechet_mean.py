"""Incremental frechet mean estimator."""
from sklearn.base import BaseEstimator


class IncrementalFrechetMean(BaseEstimator):
    r"""Incremental Frechet Mean Estimator.

    Incremental frechet mean estimator calculates sample frechet mean by
    moving iteratively along the geodesic between current mean estimate
    and next point.

    .. math::
        \text{Initialization}: m_{1} := X_{1}

     \text{Update}:  \text{Let } $\gamma_k$ \text{ be geodesic joining }
      m_{k-1}\text{ and } X_{k} \text{ then }
      m_{k} := \gamma(1/k) \,\, \forall  2 \leq k \leq N

    Asymptotic convergence to population frechet mean is guranteed for
    simply connected, complete and non-positively curved Riemannian manifolds.
    It is important to note that estimator obtained by such iterative fashion
    need not necessarily be solution to the following optimization problem.

    .. math::
        \max_{q \in M} \sum_{i=1}^{N} d(q, X_{i})^2

    where d is the riemannian metric. Also, Estimator is not permutation
    invariant , i.e.,the estimate might depend on the order in which
    incremental updates are performed.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    verbose : bool
        Verbose option.
        Optional, default: False.
    clean_state : bool
        If keeping track of last iteration or clean state of estimator.

    Notes
    -----
    * Required metric methods: `geodesic`.

    References
    ----------
    .. [CHSV2016] Cheng, Ho, Salehian, Vemuri.
        "Recursive Computation of the Frechet Mean on Non-Positively
        Curved Riemannian Manifolds with Applications",
        Riemannian Computing in Computer Vision pp 21-43, 2016.
        https://link.springer.com/chapter/10.1007/978-3-319-22957-7_2
    """

    def __init__(
        self,
        space,
        verbose=False,
        clean_state=True,
    ):
        self.space = space
        self.verbose = verbose
        self.clean_state = clean_state

        self.iter = 0
        self.estimate_ = None

    def fit(self, X, y=None, init=None):
        """Compute the incremental Frechet mean.

        Parameters
        ----------
        X : array-like, shape=[n_samples, {dim, [n, n]}]
            Training input samples.
        y : None
            Ignored.
        init : array-like, shape=[{dim, [n, n]}]
            If not None, starts mean computation from init, could be useful
            when data comes in streaming setting.
            Optional, default: None.

        Returns
        -------
        self : object
            Returns self.
        """
        N = X.shape[0]
        if init is not None:
            m_curr = init
            idxs = range(N)
        else:
            m_curr = X[0]
            self.iter += 1
            idxs = range(1, N)

        for i in idxs:
            geod_func = self.space.metric.geodesic(m_curr, X[i])
            m_curr = geod_func(1 / (self.iter + 1))
            self.iter += 1

        if self.clean_state:
            self.iter = 0

        self.estimate_ = m_curr[0]
        return self
