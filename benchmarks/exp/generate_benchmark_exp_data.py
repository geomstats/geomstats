from itertools import chain, product

import numpy as np
import pandas as pd


def spd_manifold_params(n_samples):
    manifold = "SPDMatrices"
    manifold_args = [(2,), (5,), (10,)]
    kwargs = {}
    module = "geomstats.geometry.spd_matrices"

    def affine_metric_params():
        params = []
        metric = "SPDMetricAffine"
        power_args = [-0.5, 1, 0.5]
        metric_args = list(product([item for item, in manifold_args], power_args))
        manifold_args_re = [
            item for item in manifold_args for i in range(len(power_args))
        ]
        common = (manifold, module, metric, n_samples, kwargs)
        for i in range(len(manifold_args)):
            params += [common + (manifold_args_re[i], metric_args[i])]
        return params

    def bures_wasserstein_metric_params():
        params = []
        metric = "SPDMetricBuresWasserstein"
        metric_args = manifold_args
        common = (manifold, module, metric, n_samples, kwargs)
        for i in range(len(manifold_args)):
            params += [common + (manifold_args[i], metric_args[i])]
        return params

    def euclidean_metric_params():
        params = []
        metric = "SPDMetricEuclidean"
        power_args = [-0.5, 1, 0.5]
        metric_args = list(product([item for item, in manifold_args], power_args))
        manifold_args_re = [
            item for item in manifold_args for i in range(len(power_args))
        ]
        common = (manifold, module, metric, n_samples, kwargs)
        for i in range(len(manifold_args)):
            params += [common + (manifold_args_re[i], metric_args[i])]
        return params

    def log_euclidean_metric_params():
        params = []
        metric = "SPDMetricLogEuclidean"
        metric_args = manifold_args
        common = (manifold, module, metric, n_samples, kwargs)
        for i in range(len(manifold_args)):
            params += [common + (manifold_args[i], metric_args[i])]
        return params

    return [
        bures_wasserstein_metric_params(),
        affine_metric_params(),
        euclidean_metric_params(),
        log_euclidean_metric_params(),
    ]


def stiefel_params(n_samples):
    manifold = "Stiefel"
    manifold_args = [(2, 2), (3, 3), (5, 5)]
    kwargs = {}
    module = "geomstats.geometry.stiefel"

    def stiefel_canonical_metric_params():
        params = []
        metric = "StiefelCanonicalMetric"
        metric_args = manifold_args
        common = (manifold, module, metric, n_samples, kwargs)
        for i in range(len(manifold_args)):
            params += [common + (manifold_args[i], metric_args[i])]
        return params

    return [stiefel_canonical_metric_params()]


def preshape_params(n_samples):
    manifold = "PreShape"
    manifold_args = [(3, 3), (5, 5)]
    kwargs = {}
    module = "geomstats.geometry.pre_shape"

    def pre_shape_metric_params():
        params = []
        metric = "PreShapeMetric"
        metric_args = manifold_args
        common = (manifold, module, metric, n_samples, kwargs)
        for i in range(len(manifold_args)):
            params += [common + (manifold_args[i], metric_args[i])]
        return params

    return [pre_shape_metric_params()]


def positive_lower_triangular_matrices_params(n_samples):
    manifold = "PositiveLowerTriangularMatrices"
    manifold_args = [(3,), (5,)]
    kwargs = {}
    module = "geomstats.geometry.positive_lower_triangular_matrices"

    def cholesky_metric_params():
        params = []
        metric = "CholeskyMetric"
        metric_args = manifold_args
        common = (manifold, module, metric, n_samples, kwargs)
        for i in range(len(manifold_args)):
            params += [common + (manifold_args[i], metric_args[i])]
        return params

    return [cholesky_metric_params()]


def minkowski_params(n_samples):
    manifold = "PositiveLowerTriangularMatrices"
    manifold_args = [(3,), (5,)]
    kwargs = {}
    module = "geomstats.geometry.minkowski"

    def minkowski_metric_params():
        params = []
        metric = "MinkowskiMetric"
        metric_args = manifold_args
        common = (manifold, module, metric, n_samples, kwargs)
        for i in range(len(manifold_args)):
            params += [common + (manifold_args[i], metric_args[i])]
        return params

    return [minkowski_metric_params()]


def matrices_params(n_samples):
    manifold = "Matrices"
    manifold_args = [(3,), (5,)]
    kwargs = {}
    module = "geomstats.geometry.matrices"

    def matrices_metric_params():
        params = []
        metric = "Matrices"
        metric_args = manifold_args
        common = (manifold, module, metric, n_samples, kwargs)
        for i in range(len(manifold_args)):
            params += [common + (manifold_args[i], metric_args[i])]
        return params

    return [matrices_metric_params()]


def hypersphere_params(n_samples):
    manifold = "Matrices"
    manifold_args = [(3,), (5,)]
    kwargs = {}
    module = "geomstats.geometry.hypersphere"

    def hypersphere_metric_params():
        params = []
        metric = "Hypersphere"
        metric_args = manifold_args
        common = (manifold, module, metric, n_samples, kwargs)
        for i in range(len(manifold_args)):
            params += [common + (manifold_args[i], metric_args[i])]
        return params

    return [hypersphere_metric_params()]


def grassmanian_params(n_samples):
    manifold = "Grassmannian"
    manifold_args = [(3, 3), (5, 5)]
    kwargs = {}
    module = "geomstats.geometry.grassmanian"

    def grassmannian_metric_params():
        params = []
        metric = "GrassmannianCanonicalMetric"
        metric_args = manifold_args
        common = (manifold, module, metric, n_samples, kwargs)
        for i in range(len(manifold_args)):
            params += [common + (manifold_args[i], metric_args[i])]
        return params

    return [grassmannian_metric_params()]


def full_rank_correlation_matrices_params(n_samples):
    manifold = "FullRankCorrelationMatrices"
    manifold_args = [(3,), (5,)]
    kwargs = {}
    module = "geomstats.geometry.full_rank_correlation_matrices"

    def full_rank_correlation_matrices_metric_params():
        params = []
        metric = "FullRankCorrelationAffineQuotientMetric"
        metric_args = manifold_args
        common = (manifold, module, metric, n_samples, kwargs)
        for i in range(len(manifold_args)):
            params += [common + (manifold_args[i], metric_args[i])]
        return params

    return [full_rank_correlation_matrices_metric_params()]


def generate_benchmark_exp_params(
    manifold="positive_lower_triangular_matrices", n_samples=10
):
    params_list = globals()[manifold + "_params"](n_samples)
    params_list = list(chain(*params_list))
    df = pd.DataFrame(
        params_list,
        columns=[
            "manifold",
            "metric",
            "manifold_args",
            "metric_args",
            "exp_kwargs",
            "module",
            "n_samples",
        ],
    )
    df.to_pickle("benchmark_exp.pkl")


generate_benchmark_exp_params()
