from itertools import product

import numpy as np
import pandas as pd


def spd_manifold_params(n_samples):
    manifold = "SPDManifold"
    manifold_args = [(2,), (5,), (10,)]
    kwargs = {}
    module = "geomstats.geometry.spd_matrices"

    def affine_metric_params():
        params = []
        metric = "SPDAffineMetric"
        power_args = [-0.5, 1, 0.5]
        metric_args = list(product(manifold_args, power_args))
        manifold_args_re = np.repeat(manifold_args, len(power_args)).tolist()
        for i in range(len(manifold_args)):
            params += [
                (manifold, metric, manifold_args_re[i], metric_args[i], kwargs, module)
            ]
        return params

    def bures_wasserstein_metric_params():
        params = []
        metric = "SPDMetricBuresWasserstein"
        metric_args = manifold_args
        for i in range(len(manifold_args)):
            params += [(manifold, metric, manifold_args[i], metric_args[i], kwargs)]
        return params

    def euclidean_metric_params():
        params = []
        metric = "EuclideanMetric"
        power_args = [-0.5, 1, 0.5]
        metric_args = list(product(manifold_args, power_args))
        manifold_args_re = np.repeat(manifold_args, len(power_args)).tolist()
        for i in range(len(manifold_args)):
            params += (manifold, metric, manifold_args_re[i], metric_args[i], kwargs)
        return params

    def log_euclidean_metric_params():
        params = []
        metric = "LogEuclideanMetric"
        metric_args = manifold_args
        for i in range(len(manifold_args)):
            params += [(manifold, metric, manifold_args[i], metric_args[i], kwargs)]
        return params

    return [
        bures_wasserstein_metric_params(),
        affine_metric_params(),
        euclidean_metric_params(),
        log_euclidean_metric_params(),
    ]


def stiefel_params():
    manifold = "Stiefel"
    manifold_args = [(2, 2), (3, 3), (5, 5)]
    kwargs = {}

    def stiefel_canonical_metric_params():
        params = []
        metric = "StiefelCanonicalMetric"
        metric_args = manifold_args
        for i in range(len(manifold_args)):
            params += (manifold, metric, manifold_args[i], metric_args[i], kwargs)
        return params

    return [stiefel_canonical_metric_params()]


def preshape_params():
    manifold = "PreShape"
    manifold_args = [(3, 3), (5, 5)]
    kwargs = {}

    def pre_shape_metric_params():
        params = []
        metric = "PreShapeMetric"
        metric_args = manifold_args
        for i in range(len(manifold_args)):
            params += (manifold, metric, manifold_args[i], metric_args[i], kwargs)
        return params

    return [pre_shape_metric_params()]


def positive_lower_triangular_matrices_params():
    manifold = "PositiveLowerTriangularMatrices"
    manifold_args = [(3,), (5,)]
    kwargs = {}

    def cholesky_metric_params():
        params = []
        metric = "CholeskyMetric"
        metric_args = manifold_args
        for i in range(len(manifold_args)):
            params += (manifold, metric, manifold_args[i], metric_args[i], kwargs)
        return params

    return [cholesky_metric_params()]


def minkowski_params():
    manifold = "PositiveLowerTriangularMatrices"
    manifold_args = [(3,), (5,)]
    kwargs = {}

    def minkowski_metric_params():
        params = []
        metric = "MinkowskiMetric"
        metric_args = manifold_args
        for i in range(len(manifold_args)):
            params += (manifold, metric, manifold_args[i], metric_args[i], kwargs)
        return params

    return [minkowski_metric_params()]


def matrices_params():
    manifold = "Matrices"
    manifold_args = [(3,), (5,)]
    kwargs = {}

    def matrices_metric_params():
        params = []
        metric = "Matrices"
        metric_args = manifold_args
        for i in range(len(manifold_args)):
            params += (manifold, metric, manifold_args[i], metric_args[i], kwargs)
        return params

    return [matrices_metric_params()]


def hypersphere_params():
    manifold = "Matrices"
    manifold_args = [(3,), (5,)]
    kwargs = {}

    def hypersphere_metric_params():
        params = []
        metric = "Hypersphere"
        metric_args = manifold_args
        for i in range(len(manifold_args)):
            params += (manifold, metric, manifold_args[i], metric_args[i], kwargs)
        return params

    return [hypersphere_metric_params()]


def grassmanian_params():
    manifold = "Grassmannian"
    manifold_args = [(3, 3), (5, 5)]
    kwargs = {}

    def grassmannian_metric_params():
        params = []
        metric = "GrassmannianCanonicalMetric"
        metric_args = manifold_args
        for i in range(len(manifold_args)):
            params += (manifold, metric, manifold_args[i], metric_args[i], kwargs)
        return params

    return [grassmannian_metric_params()]


def full_rank_correlation_matrices_params():
    manifold = "FullRankCorrelationMatrices"
    manifold_args = [(3,), (5,)]
    kwargs = {}

    def full_rank_correlation_matrices_metric_params():
        params = []
        metric = "FullRankCorrelationAffineQuotientMetric"
        metric_args = manifold_args
        for i in range(len(manifold_args)):
            params += (manifold, metric, manifold_args[i], metric_args[i], kwargs)
        return params

    return [full_rank_correlation_matrices_metric_params()]

def generate_benchmark_exp_params(manifold="spd_manifold", n_samples=10):
    params_fn = globals()[manifold+"_params"](n_samples)
    df = pd.DataFrame(columns=["manifold", "metric", "manifold_args", "metric_args", "exp_kwargs", "module", "n_samples" ])






